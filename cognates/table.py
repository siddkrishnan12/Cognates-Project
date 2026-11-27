"""Step 7: CSV table of labeled cognates, with the new pair as the last row.

Words are prepared once, then reversed in batches: one Lexurgy call per stage
for all words that share the same path. Pairs that stop in any step are
logged to a separate CSV with the step and reason.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

from . import dataset, reverse
from .errors import PipelineStop
from .paths import OUTPUT
from .pipeline import COLUMNS, Word, analyze, prepare_word, row

TABLE_CSV = OUTPUT / "table.csv"
SKIPPED_CSV = OUTPUT / "skipped.csv"


def _labeled_rows(pairs: list[dict], progress=print) -> tuple[list[dict], list[dict]]:
    words: dict[tuple[str, str], Word | PipelineStop] = {}
    for p in pairs:
        for lang, w in ((p["lang1"], p["word1"]), (p["lang2"], p["word2"])):
            if (lang, w) not in words:
                try:
                    words[(lang, w)] = prepare_word(w, lang)
                except PipelineStop as stop:
                    words[(lang, w)] = stop
    progress(f"prepared {len(words)} words")

    # Group reversal work by (language, path) so each stage runs once per group.
    jobs: dict[tuple[str, tuple[str, ...]], set[str]] = defaultdict(set)
    paths: dict[int, tuple[tuple[str, ...], tuple[str, ...]]] = {}
    for i, p in enumerate(pairs):
        plan = reverse.plan(p["lang1"], p["lang2"])
        paths[i] = (tuple(plan.path1), tuple(plan.path2))
        jobs[(p["lang1"], paths[i][0])].add(p["word1"])
        jobs[(p["lang2"], paths[i][1])].add(p["word2"])

    cands: dict[tuple[str, str, tuple[str, ...]], set[str] | PipelineStop] = {}
    for (lang, path), ws in jobs.items():
        ready = {w: words[(lang, w)].ipa for w in ws if isinstance(words[(lang, w)], Word)}
        try:
            result = reverse.reconstruct_many(ready, list(path))
        except PipelineStop as stop:
            result = {w: stop for w in ready}
        for w, c in result.items():
            cands[(lang, w, path)] = c
        progress(f"reversed {len(ready)} {lang} words through {len(path)} stages")

    rows, skipped = [], []
    for i, p in enumerate(pairs):
        w1, w2 = words[(p["lang1"], p["word1"])], words[(p["lang2"], p["word2"])]
        try:
            for w in (w1, w2):
                if isinstance(w, PipelineStop):
                    raise w
            c1 = cands[(p["lang1"], p["word1"], paths[i][0])]
            c2 = cands[(p["lang2"], p["word2"], paths[i][1])]
            for c in (c1, c2):
                if isinstance(c, PipelineStop):
                    raise c
            rows.append(row(w1, w2, c1, c2))
        except PipelineStop as stop:
            skipped.append({**p, "step": stop.step, "reason": stop.reason})
    return rows, skipped


PRODUCT_CSV = OUTPUT / "product.csv"


def read_word_list(path: Path) -> tuple[str, list[str]]:
    """A word-list file: one noun per line, language code as the file extension.

    `list1.en` holds English nouns. Blank lines are ignored.
    Returns (language code, nouns in file order).
    """
    from .wiktionary import language

    path = Path(path)
    lang = path.suffix.lstrip(".")
    if not lang:
        raise PipelineStop(1, f"word list {path} needs a language code as its extension, e.g. list1.en")
    language(lang)
    words = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not words:
        raise PipelineStop(1, f"word list {path} is empty")
    return lang, words


def build_product(list1: Path, list2: Path, progress=print, output: Path | None = None) -> tuple[Path, int, int]:
    """Score every ordered pair (noun1 from list1, noun2 from list2).

    Rows follow list1's order, and within each noun1, list2's order. Pairs that stop
    in any step go to the skipped log, as in the labeled table.
    """
    lang1, words1 = read_word_list(list1)
    lang2, words2 = read_word_list(list2)
    pairs = [{"lang1": lang1, "word1": w1, "lang2": lang2, "word2": w2}
             for w1 in words1 for w2 in words2]
    progress(f"{len(words1)} {lang1} nouns x {len(words2)} {lang2} nouns = {len(pairs)} pairs")
    rows, skipped = _labeled_rows(pairs, progress)
    table_path = Path(output) if output else PRODUCT_CSV
    _write(table_path, rows, COLUMNS)
    _write(skipped_path(table_path), skipped, ["lang1", "word1", "lang2", "word2", "step", "reason"])
    return table_path, len(rows), len(skipped)


def _write(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def skipped_path(table_path: Path) -> Path:
    """The skipped-pairs log for a table: output/skipped.csv for the default table,
    else <name>_skipped.csv next to the table."""
    if table_path == TABLE_CSV:
        return SKIPPED_CSV
    return table_path.with_name(f"{table_path.stem}_skipped.csv")


def build(typed1: str, lang1: str, typed2: str, lang2: str, langs: list[str] | None = None,
          progress=print, output: Path | None = None) -> tuple[Path, int, int]:
    """Write the table CSV. The new pair must succeed; labeled pairs that stop are skipped."""
    table_path = Path(output) if output else TABLE_CSV
    new_row, _ = analyze(typed1, lang1, typed2, lang2)
    pairs = dataset.load()
    if langs:
        pairs = [p for p in pairs if p["lang1"] in langs and p["lang2"] in langs]
    rows, skipped = _labeled_rows(pairs, progress)
    _write(table_path, rows + [new_row], COLUMNS)
    _write(skipped_path(table_path), skipped, ["lang1", "word1", "lang2", "word2", "shared", "step", "reason"])
    return table_path, len(rows), len(skipped)
