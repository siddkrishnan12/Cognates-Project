"""IPA cleanup, vowel stripping and reconstruction-symbol mapping.

Vowels are segments PanPhon marks [+syllabic, -consonantal]. Stripping a vowel
segment removes its length, tone and nasal marks with it. Glides (j, w) and
syllabic consonants (r̩, n̩) stay.
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import yaml

from .errors import PipelineStop
from .paths import RULES

SYMBOLS_YAML = RULES / "symbols.yaml"

# Stress marks, syllable breaks, linking ties and Wiktionary's optional-sound parentheses.
_PROSODY = re.compile(r"[ˈˌ.‿()\-]")
# Accent marks used for tone/pitch accent (Vedic ŕ̩, á). Combining acute, grave, circumflex, double acute.
_ACCENTS = re.compile("[̀́̂̋̏]")
# Non-syllabic mark on diphthong glides (haʊ̯s); diphthongs are written without it.
_NONSYLLABIC = re.compile("̯")


@lru_cache(maxsize=1)
def _ft():
    import panphon

    return panphon.FeatureTable()


def clean(ipa: str) -> str:
    """Normalize Wiktionary IPA for rule input: drop prosody marks, keep optional sounds."""
    text = unicodedata.normalize("NFD", ipa)
    text = _PROSODY.sub("", text)
    text = _ACCENTS.sub("", text)
    text = _NONSYLLABIC.sub("", text)
    text = text.replace("̆", "")  # breve: extra-short/flap mark (Vedic ɭ̆ʱ)
    # Transliteration habits in epitran output (got-Goth): macron = long, thorn = θ
    text = text.replace("̄", "ː").replace("þ", "θ")
    text = text.replace("g", "ɡ")  # ASCII g -> IPA ɡ (U+0261), which the rules use
    return unicodedata.normalize("NFC", text.split(",")[0].split()[0] if text.strip() else "")


def segments(ipa: str) -> list[str]:
    return _ft().ipa_segs(unicodedata.normalize("NFD", ipa))


def is_vowel(segment: str) -> bool:
    fts = _ft().fts(segment)
    return bool(fts) and fts["syl"] == 1 and fts["cons"] == -1


def strip_vowels(ipa: str) -> str:
    """Remove vowel segments (with their diacritics) from PanPhon-readable IPA."""
    return "".join(s for s in segments(ipa) if not is_vowel(s))


@lru_cache(maxsize=1)
def symbol_table() -> list[tuple[str, str]]:
    table = yaml.safe_load(SYMBOLS_YAML.read_text()) or {}
    pairs = [(unicodedata.normalize("NFC", str(k)), unicodedata.normalize("NFC", str(v))) for k, v in table.items()]
    return sorted(pairs, key=lambda kv: -len(kv[0]))


def to_ipa(form: str) -> str:
    """Map reconstruction symbols to IPA, longest key first, in one left-to-right pass."""
    # Map symbols first: palatovelar acutes (ḱ ǵ) must not be mistaken for pitch accents.
    text = unicodedata.normalize("NFC", form.lstrip("*"))
    out, i = [], 0
    table = symbol_table()
    while i < len(text):
        for key, value in table:
            if text.startswith(key, i):
                out.append(value)
                i += len(key)
                break
        else:
            out.append(text[i])
            i += 1
    text = unicodedata.normalize("NFD", "".join(out))
    text = _ACCENTS.sub("", text)
    text = text.replace("̄", "ː").replace("̆", "")  # macron = long, breve = short
    return unicodedata.normalize("NFC", text)


def validate(ipa: str, step: int = 6) -> str:
    """Halt when PanPhon would silently drop any symbol."""
    nfd = unicodedata.normalize("NFD", ipa)
    if not _ft().validate_word(nfd):
        known = "".join(segments(ipa))
        raise PipelineStop(step, f"PanPhon cannot read every symbol in {ipa!r} (it would keep only {known!r})")
    return ipa
