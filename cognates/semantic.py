"""Step 3: word2vec similarity between the two English translations.

Parenthetical notes are dropped, the spaCy parser picks the head noun (the
root of the phrase), and the score is the cosine similarity of the two head
nouns' Google News 300 vectors.
"""
from __future__ import annotations

import re
from functools import lru_cache

from .errors import PipelineStop
from .paths import GENSIM

W2V_GZ = GENSIM / "word2vec-google-news-300" / "word2vec-google-news-300.gz"
W2V_KV = GENSIM / "word2vec-google-news-300.kv"

_PARENS = re.compile(r"\([^)]*\)")


@lru_cache(maxsize=1)
def _nlp():
    import spacy

    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def vectors():
    """Load the vectors, converting the download to a memory-mappable file once."""
    from gensim.models import KeyedVectors

    if not W2V_KV.exists():
        kv = KeyedVectors.load_word2vec_format(str(W2V_GZ), binary=True)
        kv.save(str(W2V_KV))
    return KeyedVectors.load(str(W2V_KV), mmap="r")


def clean(translation: str) -> str:
    """Drop parenthetical notes and surrounding punctuation."""
    text = _PARENS.sub(" ", translation)
    text = text.split(";")[0].split(",")[0]
    return " ".join(text.split()).strip(" .:")


def head_noun(translation: str) -> str:
    """The root token of the cleaned phrase, e.g. 'male parent' -> 'parent'."""
    text = clean(translation)
    if not text:
        raise PipelineStop(3, f"translation {translation!r} is empty after cleaning")
    words = text.split()
    if len(words) == 1:
        return words[0]
    doc = _nlp()(text)
    root = next(t for t in doc if t.head == t)
    return root.text


def _lookup(word: str):
    kv = vectors()
    for form in (word, word.lower(), word.capitalize()):
        if form in kv.key_to_index:
            return form
    raise PipelineStop(3, f"{word!r} is missing from the word2vec vocabulary")


def similarity(translation1: str, translation2: str) -> tuple[float, str, str]:
    """Return (cosine similarity, head noun 1, head noun 2)."""
    h1, h2 = head_noun(translation1), head_noun(translation2)
    k1, k2 = _lookup(h1), _lookup(h2)
    return float(vectors().similarity(k1, k2)), h1, h2
