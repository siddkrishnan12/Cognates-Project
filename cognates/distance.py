"""Step 6: PanPhon weighted feature edit distance between vowel-stripped roots."""
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
    dropped1: str = ""  # final segment forgiven on root1 (missing-ending rule), if any
    dropped2: str = ""

    @staticmethod
    def _shown(root: str, dropped: str) -> str:
        if not dropped:
            return root
        return "".join(ipa_mod.segments(root)[:-1]) + f"({dropped})"

    def shown1(self) -> str:
        """root1 with a forgiven final consonant in parentheses, e.g. wlf(s)."""
        return self._shown(self.root1, self.dropped1)

    def shown2(self) -> str:
        return self._shown(self.root2, self.dropped2)


def prepare(candidate: str) -> str:
    """Map reconstruction symbols to IPA, validate, then strip vowels."""
    mapped = ipa_mod.validate(ipa_mod.to_ipa(candidate))
    return ipa_mod.strip_vowels(mapped)


def _without_final(root: str) -> tuple[str, str] | None:
    """(root minus its final segment, that segment), or None for an empty root."""
    segs = ipa_mod.segments(root)
    if not segs:
        return None
    return "".join(segs[:-1]), segs[-1]


def closest(candidates1: set[str], candidates2: set[str],
            drop_final1: bool = False, drop_final2: bool = False) -> Closest:
    """Minimum weighted_feature_edit_distance over all candidate pairs.

    drop_final1 / drop_final2: the other word's language gives lemmas without their
    nominative ending (languages.yaml `lemma_ending: missing`), so one final consonant
    of this root may be dropped at no cost. Every other insertion/deletion keeps its cost.
    """
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
            options = [(r1, r2, "", "")]
            if drop_final1 and (cut := _without_final(r1)) and (cut[0] or r2):
                options.append((cut[0], r2, cut[1], ""))
            if drop_final2 and (cut := _without_final(r2)) and (cut[0] or r1):
                options.append((r1, cut[0], "", cut[1]))
            for a, b, d1, d2 in options:
                d = _dist().jt_weighted_feature_edit_distance(a, b)
                if best is None or d < best.distance:
                    best = Closest(d, r1, r2, c1, c2, d1, d2)
    if best is None:
        raise PipelineStop(6, "every candidate root of both words is empty after vowel stripping")
    return best
