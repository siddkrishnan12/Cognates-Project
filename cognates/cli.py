"""Command line: cognates index | tree | build-dataset | test | table."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .errors import PipelineStop


def _test(args) -> None:
    from .pipeline import analyze

    row, d = analyze(args.noun1, args.lang1, args.noun2, args.lang2, args.forgive_missing_endings)
    for n, w, trace in ((1, d["word1"], d["trace1"]), (2, d["word2"], d["trace2"])):
        print(f"noun{n}: {w.typed} -> lemma {w.lemma.word} | translation {w.translation!r} | IPA {w.ipa} ({w.ipa_source})")
        for t in trace:
            print(f"    reversed {t.stage}: {', '.join(t.forms[:12])}{' ...' if len(t.forms) > 12 else ''}")
    print(f"shared ancestor: {d['ancestor']}")
    print(f"Step 3 semantic similarity: {row['semantic_similarity']}")
    print(f"Step 6 phonetic distance:   {row['phonetic_distance']} ({row['closest_root1']} ~ {row['closest_root2']})")
    if args.tree:
        if row["phonetic_distance"] > args.tree_threshold:
            print(f"no tree: phonetic distance {row['phonetic_distance']:g} is above the threshold "
                  f"{args.tree_threshold:g} (set --tree-threshold to change it)")
            return
        from .latex_tree import tree_tex, write_and_compile

        tex, pdf, message = write_and_compile(tree_tex(row, d), args.tree)
        print(f"tree: wrote {tex}" + (f" and {pdf}" if pdf else "") + f" ({message})")


def _table(args) -> None:
    from .table import build

    from .table import skipped_path

    path, n, skipped = build(args.noun1, args.lang1, args.noun2, args.lang2, args.langs,
                             forgive_endings=args.forgive_missing_endings, output=args.output)
    print(f"wrote {path} ({n} labeled rows + the new pair; {skipped} pairs skipped, see {skipped_path(path).name})")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="cognates", description="Test whether two Indo-European nouns are cognates.")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("index", help="build the Wiktionary noun index from data/kaikki")
    sub.add_parser("tree", help="build tree/indo-european.json from Index Diachronica")
    sub.add_parser("build-dataset", help="extract labeled cognate pairs to output/cognate_pairs.csv")
    for name, fn in (("test", _test), ("table", _table)):
        s = sub.add_parser(name)
        s.add_argument("noun1"); s.add_argument("lang1"); s.add_argument("noun2"); s.add_argument("lang2")
        s.add_argument("--forgive-missing-endings", action="store_true",
                       help="when one language's lemmas lack nominative endings (languages.yaml "
                            "lemma_ending: missing, e.g. Sanskrit), drop one final consonant of the "
                            "other word's root at no cost (off by default)")
        if name == "test":
            s.add_argument("--tree", type=Path, metavar="PATH.tex",
                           help="when the pair scores as cognate (phonetic distance at or below "
                                "--tree-threshold), write a LaTeX tree of the shared root and both "
                                "derivation chains to PATH.tex and compile it to PDF with XeLaTeX")
            s.add_argument("--tree-threshold", type=float, default=2.0, metavar="D",
                           help="largest phonetic distance that counts as cognate for --tree (default 2.0)")
        if name == "table":
            s.add_argument("--langs", nargs="+", help="limit labeled pairs to these language codes")
            s.add_argument("--output", type=Path, metavar="PATH",
                           help="table CSV path (default output/table.csv); skipped pairs go to "
                                "<name>_skipped.csv next to it")
        s.set_defaults(fn=fn)
    args = p.parse_args(argv)
    try:
        if args.cmd == "index":
            from .wiktionary import build_index
            build_index()
        elif args.cmd == "tree":
            from .tree import main as tree_main
            tree_main()
        elif args.cmd == "build-dataset":
            from . import dataset
            pairs = dataset.build_pairs()
            dataset.save(pairs)
            print(f"wrote {len(pairs)} pairs to {dataset.PAIRS_CSV}")
        else:
            args.fn(args)
    except PipelineStop as stop:
        print(f"stopped at {stop}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
