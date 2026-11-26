"""Steps 1-2: Wiktionary lookup, lemmatization, first translation and IPA.

Noun entries from the kaikki.org JSONL files go into a SQLite index once,
keyed by headword and by romanization (Gothic and Sanskrit use native scripts).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

import yaml

from .errors import PipelineStop
from .paths import DATA, KAIKKI, ROOT

INDEX = DATA / "wiktionary.sqlite"
LANGUAGES_YAML = ROOT / "languages.yaml"
MAX_FORM_OF_HOPS = 3


@lru_cache(maxsize=1)
def languages() -> dict:
    return yaml.safe_load(LANGUAGES_YAML.read_text())


def language(code: str) -> dict:
    langs = languages()
    if code not in langs:
        supported = ", ".join(f"{k} ({v['name']})" for k, v in langs.items())
        raise PipelineStop(1, f"language {code!r} is unsupported; supported: {supported}")
    return langs[code]


def _slim(e: dict) -> dict:
    return {
        "word": e["word"],
        "lang_code": e.get("lang_code"),
        "romanizations": [f["form"] for f in e.get("forms", []) if "romanization" in f.get("tags", [])],
        "senses": [
            {"glosses": s.get("glosses", []), "tags": s.get("tags", []),
             "form_of": [f.get("word") for f in s.get("form_of", [])]}
            for s in e.get("senses", [])
        ],
        "sounds": [{"ipa": s["ipa"], "tags": s.get("tags", [])} for s in e.get("sounds", []) if "ipa" in s],
        "etymology_templates": [
            {"name": t.get("name"), "args": t.get("args", {})} for t in e.get("etymology_templates", [])
        ],
    }


def build_index(codes: list[str] | None = None) -> None:
    """Index every noun entry of the configured languages."""
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(INDEX)
    con.execute("CREATE TABLE IF NOT EXISTS noun (lang TEXT, key TEXT, entry TEXT)")
    con.execute("CREATE INDEX IF NOT EXISTS noun_key ON noun (lang, key)")
    for code in codes or list(languages()):
        con.execute("DELETE FROM noun WHERE lang = ?", (code,))
        rows = []
        with open(KAIKKI / languages()[code]["kaikki"], encoding="utf-8") as fh:
            for line in fh:
                e = json.loads(line)
                if e.get("pos") != "noun" or e.get("lang_code") != code:
                    continue
                slim = _slim(e)
                blob = json.dumps(slim, ensure_ascii=False)
                keys = {slim["word"], *slim["romanizations"]}
                rows.extend((code, k, blob) for k in keys)
                if len(rows) > 50_000:
                    con.executemany("INSERT INTO noun VALUES (?, ?, ?)", rows)
                    rows = []
        con.executemany("INSERT INTO noun VALUES (?, ?, ?)", rows)
        con.commit()
        print(f"indexed {code}")
    con.close()


@lru_cache(maxsize=1)
def _con() -> sqlite3.Connection:
    if not INDEX.exists():
        raise PipelineStop(1, f"Wiktionary index missing; run `cognates index` to build {INDEX}")
    return sqlite3.connect(INDEX, check_same_thread=False)


def entries(word: str, code: str) -> list[dict]:
    rows = _con().execute("SELECT entry FROM noun WHERE lang = ? AND key = ?", (code, word)).fetchall()
    return [json.loads(r[0]) for r in rows]


def _is_form_of(entry: dict) -> str | None:
    """The lemma an inflected-form entry points to, if every sense is a form-of sense."""
    senses = entry["senses"]
    if senses and all(s["form_of"] for s in senses):
        return senses[0]["form_of"][0]
    return None


@dataclass
class Lemma:
    typed: str
    lang: str
    entry: dict

    @property
    def word(self) -> str:
        return self.entry["word"]

    @property
    def romanization(self) -> str | None:
        return (self.entry["romanizations"] or [None])[0]


def lemmatize(word: str, code: str) -> Lemma:
    """Step 1, part 1: follow form-of links to the dictionary form."""
    language(code)
    current = word
    for _ in range(MAX_FORM_OF_HOPS + 1):
        found = entries(current, code)
        if not found:
            raise PipelineStop(1, f"no Wiktionary noun entry for {current!r} ({code})")
        lemma_entries = [e for e in found if not _is_form_of(e)]
        if lemma_entries:
            return Lemma(word, code, lemma_entries[0])
        current = _is_form_of(found[0])
    raise PipelineStop(1, f"form-of links from {word!r} ({code}) never reach a lemma")


def translation(lemma: Lemma) -> str:
    """Step 1, part 2: the first listed English translation.

    For an English noun the lemma itself is its English translation.
    """
    if lemma.lang == "en":
        return lemma.word
    for sense in lemma.entry["senses"]:
        if sense["glosses"] and not sense["form_of"]:
            return sense["glosses"][0]
    raise PipelineStop(1, f"no English translation listed for {lemma.word!r} ({lemma.lang})")


def _strip_delims(ipa: str) -> str:
    return ipa.strip().strip("/[]")


def ipa(lemma: Lemma) -> tuple[str, str]:
    """Step 2: the first phonemic IPA, else the first phonetic IPA, else epitran.

    When the language config names a `prefer_tag` (English: Received-Pronunciation),
    the first phonemic IPA carrying that tag comes first.
    Returns (ipa, source) where source is 'wiktionary-phonemic',
    'wiktionary-phonetic' or 'epitran'.
    """
    prefer = language(lemma.lang).get("prefer_tag")
    if prefer:
        for s in lemma.entry["sounds"]:
            if s["ipa"].startswith("/") and prefer in s["tags"]:
                return _strip_delims(s["ipa"]), "wiktionary-phonemic"
    sounds = [s["ipa"] for s in lemma.entry["sounds"]]
    for s in sounds:
        if s.startswith("/"):
            return _strip_delims(s), "wiktionary-phonemic"
    for s in sounds:
        if s.startswith("["):
            return _strip_delims(s), "wiktionary-phonetic"
    code = language(lemma.lang)["epitran"]
    if code is None:
        raise PipelineStop(2, f"{lemma.word!r} ({lemma.lang}) has no Wiktionary IPA, and epitran lacks the language")
    import epitran

    cfg = language(lemma.lang)
    if code.endswith("-Latn") and lemma.romanization:
        return epitran.Epitran(code).transliterate(lemma.romanization), "epitran"
    if cfg.get("epitran_native"):
        return epitran.Epitran(cfg["epitran_native"]).transliterate(lemma.word), "epitran"
    return epitran.Epitran(code).transliterate(lemma.word), "epitran"
