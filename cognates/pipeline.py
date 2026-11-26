"""Steps 1-6 for one pair, and in batches for the labeled dataset."""
from __future__ import annotations

from dataclasses import dataclass

from . import distance, ipa, reverse, semantic, wiktionary
from .errors import PipelineStop

COLUMNS = [
    "noun1", "language1", "noun2", "language2",
    "translation1", "translation2", "semantic_similarity",
    "closest_root1", "closest_root2", "phonetic_distance",
]


@dataclass
class Word:
    typed: str
    lang: str
    lemma: wiktionary.Lemma
    translation: str
    ipa: str  # cleaned, vowels intact (Step 5 strips vowels after the reversed rules)
    ipa_source: str


def prepare_word(typed: str, lang: str) -> Word:
    """Steps 1-2."""
    lemma = wiktionary.lemmatize(typed, lang)
    trans = wiktionary.translation(lemma)
    raw, source = wiktionary.ipa(lemma)
    return Word(typed, lang, lemma, trans, ipa.clean(raw), source)


def display(word: Word) -> str:
    rom = word.lemma.romanization
    return f"{word.lemma.word} ({rom})" if rom else word.lemma.word


def row(w1: Word, w2: Word, cands1: set[str], cands2: set[str], forgive_endings: bool = False) -> dict:
    """Steps 3 and 6 for two prepared words with their candidate roots.

    forgive_endings (off by default; CLI --forgive-missing-endings): when one language's
    Wiktionary lemmas lack the nominative ending (languages.yaml `lemma_ending: missing`,
    e.g. Sanskrit stems), one final consonant of the other word's root is free.
    """
    return _score(w1, w2, cands1, cands2, forgive_endings)[0]


def _score(w1: Word, w2: Word, cands1: set[str], cands2: set[str],
           forgive_endings: bool = False) -> tuple[dict, distance.Closest]:
    """row() plus the closest candidate pair behind its phonetic distance."""
    sim, _, _ = semantic.similarity(w1.translation, w2.translation)
    missing1 = forgive_endings and wiktionary.language(w1.lang).get("lemma_ending") == "missing"
    missing2 = forgive_endings and wiktionary.language(w2.lang).get("lemma_ending") == "missing"
    best = distance.closest(cands1, cands2, drop_final1=missing2 and not missing1,
                            drop_final2=missing1 and not missing2)
    return {
        "noun1": display(w1), "language1": w1.lang,
        "noun2": display(w2), "language2": w2.lang,
        "translation1": w1.translation, "translation2": w2.translation,
        "semantic_similarity": round(sim, 4),
        "closest_root1": best.shown1(), "closest_root2": best.shown2(),
        "phonetic_distance": round(best.distance, 4),
    }, best


def analyze(typed1: str, lang1: str, typed2: str, lang2: str,
            forgive_endings: bool = False) -> tuple[dict, dict]:
    """Run Steps 1-6 on one pair. Returns (table row, details for display)."""
    w1, w2 = prepare_word(typed1, lang1), prepare_word(typed2, lang2)
    plan = reverse.plan(lang1, lang2)
    cands1, trace1 = reverse.reconstruct(w1.ipa, plan.path1)
    cands2, trace2 = reverse.reconstruct(w2.ipa, plan.path2)
    details = {
        "ancestor": reverse.target_name(plan.ancestor),
        "word1": w1, "word2": w2,
        "trace1": trace1, "trace2": trace2,
        "candidates1": sorted(cands1), "candidates2": sorted(cands2),
        "path1": plan.path1, "path2": plan.path2,
    }
    result, best = _score(w1, w2, cands1, cands2, forgive_endings)
    details["closest"] = best
    return result, details


__all__ = ["COLUMNS", "PipelineStop", "Word", "analyze", "prepare_word", "row"]
