"""Step 6: PanPhon weighted feature edit distance between vowel-stripped roots.

The metric is `jt_weighted_feature_edit_distance` (team decision 2026-11-27): substitutions
cost 0.25-4.75 by feature, and every insertion or deletion costs 1.8125.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from . import ipa as ipa_mod
from .errors import PipelineStop


@lru_cache(maxsize=1)
def _dist():
    import panphon.distance

    return panphon.distance.Distance()


@dataclass
class Closest:
    distance: float
    root1: str  # vowel-stripped IPA
    root2: str
    source1: str  # candidate form before mapping and stripping
    source2: str


def prepare(candidate: str) -> str:
    """Map reconstruction symbols to IPA, validate, then strip vowels."""
    mapped = ipa_mod.validate(ipa_mod.to_ipa(candidate))
    return ipa_mod.strip_vowels(mapped)


def closest(candidates1: set[str], candidates2: set[str]) -> Closest:
    """Minimum jt_weighted_feature_edit_distance over all candidate pairs."""
    if not candidates1 or not candidates2:
        raise PipelineStop(6, "a word has no candidate roots")
    # Sorted, so ties at the minimum resolve the same way on every run.
    roots1 = {c: prepare(c) for c in sorted(candidates1)}
    roots2 = {c: prepare(c) for c in sorted(candidates2)}
    best: Closest | None = None
    for c1, r1 in roots1.items():
        for c2, r2 in roots2.items():
            if not r1 and not r2:
                continue
            d = _dist().jt_weighted_feature_edit_distance(r1, r2)
            if best is None or d < best.distance:
                best = Closest(d, r1, r2, c1, c2)
    if best is None:
        raise PipelineStop(6, "every candidate root of both words is empty after vowel stripping")
    return best
