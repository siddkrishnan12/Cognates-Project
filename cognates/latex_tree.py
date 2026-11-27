"""A LaTeX tree for a cognate pair: the shared root on top, each word's chain below.

The root node shows the shared consonant skeleton (Step 6's closest roots) and the
ancestor language. Each branch starts with its own full reconstructed form at the
ancestor, then lists one derivation stage by stage, labeled with each stage's language,
down to the attested word, its IPA and its language variant. The file compiles with
XeLaTeX (fontspec): Charis SIL for Latin and IPA, Devanagari MT and Noto Sans Gothic
for those scripts.
"""
from __future__ import annotations

import shutil
import subprocess
import unicodedata
from pathlib import Path

from . import reverse
from .pipeline import display
from .wiktionary import language

_PREAMBLE = r"""\documentclass[border=10pt]{standalone}
\usepackage{fontspec}
\setmainfont{CharisSIL}[Extension=.ttf, UprightFont=*-Regular, BoldFont=*-Bold,
  ItalicFont=*-Italic, BoldItalicFont=*-BoldItalic]
\newfontfamily\devafont{Devanagari MT}[Script=Devanagari]
\newfontfamily\gothfont{Noto Sans Gothic}
\usepackage{forest}
\begin{document}
"""

_SPECIAL = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "%": r"\%", "$": r"\$",
            "&": r"\&", "#": r"\#", "_": r"\_", "^": r"\textasciicircum{}", "~": r"\textasciitilde{}"}


def _script(ch: str) -> str | None:
    cp = ord(ch)
    if 0x0900 <= cp <= 0x097F or 0x1CD0 <= cp <= 0x1CFF or 0xA8E0 <= cp <= 0xA8FF:
        return "deva"
    if 0x10330 <= cp <= 0x1034F:
        return "goth"
    return None


def tex_text(text: str) -> str:
    """Escape LaTeX specials and wrap Devanagari and Gothic runs in their fonts.

    Combining marks stay with the run of the character they follow.
    """
    out: list[str] = []
    runs: list[tuple[str | None, list[str]]] = []
    for ch in unicodedata.normalize("NFC", text):
        script = runs[-1][0] if runs and unicodedata.category(ch).startswith("M") else _script(ch)
        if runs and runs[-1][0] == script:
            runs[-1][1].append(ch)
        else:
            runs.append((script, [ch]))
    for script, chars in runs:
        body = "".join(_SPECIAL.get(c, c) for c in chars)
        out.append({"deva": r"{\devafont " + body + "}", "goth": r"{\gothfont " + body + "}"}.get(script, body))
    return "".join(out)


def _node(*lines: str) -> str:
    return "{" + r"\\ ".join(lines) + "}"


def _branch(word, final_form: str, trace: list, ancestor: str) -> str:
    """Nested forest nodes for one word, from its ancestor form down to the word."""
    steps = reverse.chain(final_form, trace, word.ipa) if trace else [(None, word.ipa)]
    variant = reverse.target_name(language(word.lang)["stage"])
    nodes: list[str] = []
    for stage, form in steps:
        if stage is None:
            nodes.append(_node(tex_text(display(word)), tex_text(f"/{word.ipa}/"),
                               r"\textsc{" + tex_text(variant) + "}"))
        else:
            lang = reverse.nodes()[stage].source or ancestor
            nodes.append(_node("*" + tex_text(form), r"\textsc{" + tex_text(lang) + "}"))
    # nest: [a [b [c]]]
    tree = ""
    for n in reversed(nodes):
        tree = f"[{n} {tree}]" if tree else f"[{n}]"
    return tree


def tree_tex(row: dict, details: dict) -> str:
    """The full LaTeX document for one analyzed pair (pipeline.analyze output)."""
    best = details["closest"]
    ancestor = details["ancestor"]
    r1, r2 = best.root1, best.root2
    skeleton = "*" + tex_text(r1) if r1 == r2 else "*" + tex_text(r1) + " ~ *" + tex_text(r2)
    root = _node(skeleton, r"\textsc{" + tex_text(ancestor) + "}",
                 r"{\footnotesize phonetic distance " + f"{row['phonetic_distance']:g}" +
                 r", semantic similarity " + f"{row['semantic_similarity']:g}" + "}")
    b1 = _branch(details["word1"], best.source1, details["trace1"], ancestor)
    b2 = _branch(details["word2"], best.source2, details["trace2"], ancestor)
    body = (r"\begin{forest}" "\n"
            r"for tree={align=center, parent anchor=south, child anchor=north, l sep=7mm, s sep=14mm}" "\n"
            f"[{root}\n  {b1}\n  {b2}\n]\n"
            r"\end{forest}" "\n")
    return _PREAMBLE + body + r"\end{document}" + "\n"


def write_and_compile(tex: str, path: Path) -> tuple[Path, Path | None, str]:
    """Write the .tex file and compile it with XeLaTeX.

    Returns (tex path, pdf path or None, message). Without XeLaTeX, or when compiling
    fails, the .tex file stays and the message says why.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tex, encoding="utf-8")
    xelatex = shutil.which("xelatex")
    if not xelatex:
        return path, None, "XeLaTeX not found; compile the .tex file with xelatex to get a PDF"
    proc = subprocess.run([xelatex, "-interaction=nonstopmode", "-halt-on-error", path.name],
                          cwd=path.parent, capture_output=True, text=True)
    pdf = path.with_suffix(".pdf")
    log = path.with_suffix(".log")
    if proc.returncode != 0 or not pdf.exists():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-15:]) if log.exists() else proc.stdout[-1500:]
        return path, None, f"XeLaTeX failed; see {log}\n{tail}"
    for ext in (".aux", ".log"):
        path.with_suffix(ext).unlink(missing_ok=True)
    return path, pdf, "compiled with XeLaTeX"
