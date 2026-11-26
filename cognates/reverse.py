"""Steps 4-5: run reversed stage rules through Lexurgy back to the nearest shared ancestor.

Each stage has a reversed rule file `rules/<family>/<stage-id>.lsc` and an
optional merger sidecar `<stage-id>.mergers.yaml`:

    - sound: a
      sources: [a, e, o]   # tag ¹ = a, ² = e, ³ = o

Before a stage runs, every occurrence of a merged sound gets one floating tag
per source, independently per position, and each combination becomes one input
line. Rules ignore floating tags unless they ask for one (e.g. `a² => e`), so
later reversed rules still see the plain sound. A generated final rule clears
leftover tags, and duplicate outputs are dropped before the next stage.
"""
from __future__ import annotations

import itertools
import re
import subprocess
import tempfile
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from . import ipa as ipa_mod
from . import tree as tree_mod
from .errors import PipelineStop
from .paths import LEXURGY, RULES
from .wiktionary import language

FAMILY_DIR = RULES / "indo-european"
TAGS = "¹²³⁴⁵⁶⁷⁸⁹"
MAX_VARIANTS = 20_000


@lru_cache(maxsize=1)
def nodes() -> dict[str, tree_mod.Node]:
    return tree_mod.load()


def target_name(stage: str | None) -> str:
    return tree_mod.ROOT_NAME if stage is None else nodes()[stage].target


def path_to(lang: str, stop_at: str | None) -> list[str]:
    """Stage ids to reverse, youngest first, from a language up to (not including) stop_at."""
    cfg = language(lang)
    lineage = tree_mod.lineage(nodes(), cfg["stage"])
    processes = {nodes()[p].parent: p for p in cfg.get("processes", [])}
    out = []
    for s in lineage:
        if s == stop_at:
            break
        if s in processes:  # a side-branch process happened after its parent stage
            out.append(processes[s])
        out.append(s)
    return out


@dataclass
class Plan:
    ancestor: str | None  # stage id whose target is the shared ancestor; None = PIE
    path1: list[str]
    path2: list[str]


def plan(lang1: str, lang2: str) -> Plan:
    s1, s2 = language(lang1)["stage"], language(lang2)["stage"]
    anc = tree_mod.nearest_common_ancestor(nodes(), s1, s2)
    # Direct descent: one language is the other's ancestor, so only the younger reverses.
    return Plan(anc, path_to(lang1, anc), path_to(lang2, anc))


# ---------------------------------------------------------------- Lexurgy runs


def _stage_files(stage: str) -> tuple[Path, list[dict]]:
    lsc = FAMILY_DIR / f"{stage}.lsc"
    if not lsc.exists():
        raise PipelineStop(5, f"reversed rules for stage {stage!r} ({nodes()[stage].title}) are not drafted yet")
    sidecar = FAMILY_DIR / f"{stage}.mergers.yaml"
    mergers = yaml.safe_load(sidecar.read_text()) if sidecar.exists() else []
    return lsc, mergers or []


def _untagged(segment: str) -> str:
    return segment.rstrip(TAGS)


def tag_variants(form: str, mergers: list[dict]) -> list[str]:
    """Every per-position assignment of merger tags to one form.

    A sidecar entry `optional_ending: z` reverses a word-final deletion: each
    variant also appears with that ending added back.
    """
    endings = [unicodedata.normalize("NFC", m["optional_ending"]) for m in mergers if "optional_ending" in m]
    mergers = [m for m in mergers if "sound" in m]
    variants = _tag_positions(form, mergers)
    return variants + [v + e for e in endings for v in variants]


def _applies(m: dict, segs: list[str], i: int) -> bool:
    """Context keys limit where an entry tags: before/after (segment lists), initial, final."""
    nxt = _untagged(segs[i + 1]) if i + 1 < len(segs) else None
    prv = _untagged(segs[i - 1]) if i > 0 else None
    if "before" in m and nxt not in m["before"]:
        return False
    if "after" in m and prv not in m["after"]:
        return False
    if m.get("initial") and i != 0:
        return False
    if m.get("final") and i != len(segs) - 1:
        return False
    return True


def _tag_positions(form: str, mergers: list[dict]) -> list[str]:
    """Tag each merged sound with one tag per source of the first entry whose context fits."""
    entries: dict[str, list[dict]] = {}
    for m in mergers:
        m = {**m, "before": [unicodedata.normalize("NFC", s) for s in m["before"]]} if "before" in m else m
        m = {**m, "after": [unicodedata.normalize("NFC", s) for s in m["after"]]} if "after" in m else m
        entries.setdefault(unicodedata.normalize("NFC", m["sound"]), []).append(m)
    segs = _lexurgy_segments(form)
    choices = []
    for i, seg in enumerate(segs):
        match = next((m for m in entries.get(_untagged(seg), []) if _applies(m, segs, i)), None)
        n = len(match["sources"]) if match else 0
        choices.append([seg + TAGS[k] for k in range(n)] if n else [seg])
    total = 1
    for c in choices:
        total *= len(c)
    if total > MAX_VARIANTS:
        raise PipelineStop(5, f"{form!r} branches into {total} variants (limit {MAX_VARIANTS})")
    return ["".join(p) for p in itertools.product(*choices)]


def _lexurgy_segments(form: str) -> list[str]:
    """PanPhon segments, falling back to single characters for symbols PanPhon lacks."""
    text = unicodedata.normalize("NFC", form)
    segs = ipa_mod.segments(text)
    joined = unicodedata.normalize("NFC", "".join(segs))
    if joined == text:
        return [unicodedata.normalize("NFC", s) for s in segs]
    # Keep unknown symbols (e.g. h₂, H) as their own characters.
    out, i = [], 0
    nfc_segs = [unicodedata.normalize("NFC", s) for s in segs]
    j = 0
    while i < len(text):
        if j < len(nfc_segs) and text.startswith(nfc_segs[j], i):
            out.append(nfc_segs[j])
            i += len(nfc_segs[j])
            j += 1
        else:
            out.append(text[i])
            i += 1
    return out


_DECLARED = re.compile(r"^\s*Symbol\s+(.+)$", re.M)


def _header(words: list[str], stage_text: str) -> str:
    declared = set()
    for line in _DECLARED.findall(stage_text):
        declared.update(s.strip() for s in line.split(",") if s.strip())
    multi = set()
    for w in words:
        for seg in _lexurgy_segments(w):
            base = _untagged(seg)
            if len(base) > 1 and base not in declared:
                multi.add(base)
    lines = [f"Feature src(*s0, {', '.join(f's{i + 1}' for i in range(len(TAGS)))})"]
    lines += [f"Diacritic {t} (floating) [s{i + 1}]" for i, t in enumerate(TAGS)]
    if multi:
        lines.append("Symbol " + ", ".join(sorted(multi)))
    return "\n".join(lines) + "\n\n"


_UNTAG = "clear-merger-tags:\n" + "\n".join(f"    [s{i + 1}] => [s0]" for i in range(len(TAGS))) + "\n"


def run_stage_many(batch: dict[str, set[str]], stage: str,
                   links: dict[str, dict[str, set[str]]] | None = None) -> dict[str, set[str]]:
    """Reverse one stage for many words in a single Lexurgy call.

    batch maps a key (any id) to its current candidate forms. Words that branch
    past MAX_VARIANTS map to a PipelineStop instead of a set. When `links` is
    given, links[key][output] collects the input forms that produced each output
    (Lexurgy writes one output line per input line).
    """
    lsc, mergers = _stage_files(stage)
    lines: list[tuple[str, str, str]] = []  # (key, source form, tagged variant)
    failed: dict[str, PipelineStop] = {}
    for key, forms in batch.items():
        try:
            per_key = sorted({(f, v) for f in forms for v in tag_variants(f, mergers)})
        except PipelineStop as stop:
            failed[key] = stop
            continue
        lines.extend((key, f, v) for f, v in per_key)
    out: dict[str, set[str]] = {k: set() for k in batch if k not in failed}
    if lines:
        stage_text = lsc.read_text()
        with tempfile.TemporaryDirectory() as tmp:
            rules = Path(tmp) / "stage.lsc"
            rules.write_text(_header([w for _, _, w in lines], stage_text) + stage_text.rstrip() + "\n\n" + _UNTAG)
            wli = Path(tmp) / "words.wli"
            wli.write_text("\n".join(w for _, _, w in lines) + "\n")
            proc = subprocess.run([str(LEXURGY), "sc", str(rules), str(wli)], capture_output=True, text=True)
            ev = Path(tmp) / "words_ev.wli"
            if proc.returncode != 0 or not ev.exists():
                raise PipelineStop(5, f"Lexurgy failed on stage {stage!r}: {(proc.stderr or proc.stdout).strip()[:500]}")
            results = ev.read_text().splitlines()
        if len(results) != len(lines):
            raise PipelineStop(5, f"Lexurgy returned {len(results)} lines for {len(lines)} inputs on stage {stage!r}")
        for (key, src, _), res in zip(lines, results):
            res = unicodedata.normalize("NFC", res.strip())
            if res:
                out[key].add(res)
                if links is not None:
                    links.setdefault(key, {}).setdefault(res, set()).add(src)
    return {**out, **failed}


def run_stage(forms: set[str], stage: str,
              links: dict[str, set[str]] | None = None) -> set[str]:
    """Reverse one stage for one word. `links`, if given, collects output -> input forms."""
    all_links: dict[str, dict[str, set[str]]] | None = {} if links is not None else None
    result = run_stage_many({"w": forms}, stage, all_links)["w"]
    if isinstance(result, PipelineStop):
        raise result
    if links is not None:
        links.update(all_links.get("w", {}))
    return result


@dataclass
class Trace:
    stage: str
    forms: list[str]
    parents: dict[str, list[str]] = field(default_factory=dict)  # output form -> input forms


def reconstruct(form: str, path: list[str]) -> tuple[set[str], list[Trace]]:
    """Reverse one cleaned IPA form through the given stages (youngest first)."""
    forms = {unicodedata.normalize("NFC", form)}
    trace = []
    for stage in path:
        links: dict[str, set[str]] = {}
        forms = run_stage(forms, stage, links)
        trace.append(Trace(stage, sorted(forms), {k: sorted(v) for k, v in links.items()}))
    return forms, trace


def chain(final_form: str, trace: list[Trace], start_form: str) -> list[tuple[str | None, str]]:
    """One derivation from a final candidate back to the input form.

    Returns [(stage, form), ...] from the oldest form down to the input. Each form is
    the output of reversing `stage`; the input form has stage None. Where several input
    forms produced the same output, the first in sorted order is taken.
    """
    steps: list[tuple[str | None, str]] = []
    form = unicodedata.normalize("NFC", final_form)
    for t in reversed(trace):
        if form not in t.parents:
            raise PipelineStop(5, f"no recorded derivation for {form!r} at stage {t.stage!r}")
        steps.append((t.stage, form))
        form = t.parents[form][0]
    steps.append((None, form))
    if form != unicodedata.normalize("NFC", start_form):
        raise PipelineStop(5, f"derivation ended at {form!r}, not the input {start_form!r}")
    return steps


def reconstruct_many(forms: dict[str, str], path: list[str]) -> dict[str, set[str] | PipelineStop]:
    """Reverse many forms (key -> cleaned IPA) through the same stages."""
    state: dict[str, set[str] | PipelineStop] = {k: {unicodedata.normalize("NFC", f)} for k, f in forms.items()}
    for stage in path:
        live = {k: v for k, v in state.items() if isinstance(v, set)}
        state.update(run_stage_many(live, stage))
    return state
