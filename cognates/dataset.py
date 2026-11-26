"""Step 7: labeled cognate pairs from Wiktionary etymologies.

Two nouns in different supported languages count as cognates when their
inherited (`inh`) links reach the same ancestor form. Nouns with any borrowing
template, or any compound/affix template anywhere in their etymology section
(which also describes how their ancestors were formed), are left out.
"""
from __future__ import annotations

import csv
import itertools
import json
import sqlite3
import unicodedata
from collections import defaultdict

from .paths import OUTPUT
from .wiktionary import INDEX, languages

PAIRS_CSV = OUTPUT / "cognate_pairs.csv"

INHERIT = {"inh", "inh+"}
BORROW = {"bor", "bor+", "lbor", "ubor", "slbor", "obor", "calque", "cal", "clq", "pcal", "translit", "sl"}
COMPOUND = {"compound", "com", "com+", "affix", "af", "prefix", "pre", "suffix", "suf", "confix", "con", "surf"}
ETY_TEMPLATES = {"ety", "etymon"}
ETY_INHERIT = {"inh"}
ETY_BORROW = {"bor", "lbor", "ubor", "slbor", "calque", "clq", "cal", "pcal", "translit"}
ETY_COMPOUND = {"af", "afeq", "com", "compound", "affix", "suf", "pre", "confix"}


def _norm(form: str) -> str:
    return unicodedata.normalize("NFC", form.strip())


def _ety_keyword(value: str) -> str:
    return value[1:].split("<", 1)[0]


def classify(entry: dict) -> tuple[set[tuple[str, str]], bool, bool]:
    """Return (inherited ancestors as (lang, form), has borrowing, has compound/affix)."""
    ancestors: set[tuple[str, str]] = set()
    borrowed = compound = False
    for t in entry["etymology_templates"]:
        name, args = t["name"], t["args"]
        if name in INHERIT and args.get("3"):
            ancestors.add((args.get("2", ""), _norm(args["3"])))
        elif name in BORROW:
            borrowed = True
        elif name in COMPOUND:
            compound = True
        elif name in ETY_TEMPLATES:
            keyword = None
            for k in sorted((k for k in args if k.isdigit()), key=int):
                v = str(args[k])
                if v.startswith(":"):
                    keyword = _ety_keyword(v)
                    borrowed |= keyword in ETY_BORROW
                    compound |= keyword in ETY_COMPOUND
                elif keyword in ETY_INHERIT and ":" in v:
                    lang, form = v.split(":", 1)
                    ancestors.add((lang, _norm(form.split("<", 1)[0])))
    return ancestors, borrowed, compound


def build_pairs(langs: list[str] | None = None) -> list[dict]:
    """All cross-language cognate pairs among clean nouns, keyed by a shared ancestor form."""
    langs = langs or list(languages())
    con = sqlite3.connect(INDEX)
    by_ancestor: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for lang in langs:
        seen = set()
        for (blob,) in con.execute("SELECT entry FROM noun WHERE lang = ?", (lang,)):
            entry = json.loads(blob)
            if entry["word"] in seen:
                continue
            seen.add(entry["word"])
            if entry["senses"] and all(s["form_of"] for s in entry["senses"]):
                continue
            ancestors, borrowed, compound = classify(entry)
            if borrowed or compound:
                continue
            for anc in ancestors:
                by_ancestor[anc].add((lang, entry["word"]))
    con.close()

    pairs: dict[tuple, dict] = {}
    for (anc_lang, anc_form), words in by_ancestor.items():
        for (l1, w1), (l2, w2) in itertools.combinations(sorted(words), 2):
            if l1 == l2:
                continue
            key = (l1, w1, l2, w2)
            pairs.setdefault(key, {"lang1": l1, "word1": w1, "lang2": l2, "word2": w2, "shared": set()})
            pairs[key]["shared"].add(f"{anc_lang}:{anc_form}")
    out = []
    for p in pairs.values():
        p["shared"] = "; ".join(sorted(p["shared"]))
        out.append(p)
    return sorted(out, key=lambda p: (p["lang1"], p["lang2"], p["word1"], p["word2"]))


def save(pairs: list[dict]) -> None:
    PAIRS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(PAIRS_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["lang1", "word1", "lang2", "word2", "shared"])
        writer.writeheader()
        writer.writerows(pairs)


def load() -> list[dict]:
    with open(PAIRS_CSV, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
