"""Build the Indo-European stage tree from Index Diachronica item 17.

Each section titled "X to Y" becomes a stage node whose target language is Y.
A stage's parent is the stage whose target is X, searched first among the
nodes closest in the table-of-contents nesting. When no title matches, the
bullet parent is used. Sections without "to" in their title either group
other sections (containers, skipped) or describe a single process. A process
becomes a dead-end side branch under its bullet parent, and a language uses it
only when the language config lists it.
"""
from __future__ import annotations

import html
import json
import re
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path

from .paths import DATA, TREE

SOURCE_HTML = DATA / "diachronica" / "all.html"
TREE_JSON = TREE / "indo-european.json"
ROOT_NAME = "Proto-Indo-European"
FAMILY_ITEM = "17"


@dataclass
class Node:
    id: str
    number: str
    title: str
    source: str | None
    target: str | None
    bullet_parent: str | None
    parent: str | None = None
    kind: str = "stage"  # stage | process | container
    attribution: str = ""
    rules: list[str] = field(default_factory=list)
    merged: list[str] = field(default_factory=list)


class _TocParser(HTMLParser):
    """Collect (id, title, bullet_parent) for every link inside item 17's list."""

    def __init__(self):
        super().__init__()
        self.in_family = False
        self.depth = 0  # <ol> depth relative to the family item
        self.stack: list[str | None] = []  # last link id seen at each depth
        self.items: list[tuple[str, str, str | None]] = []
        self._href: str | None = None
        self._text: list[str] = []
        self.done = False

    def handle_starttag(self, tag, attrs):
        if self.done:
            return
        a = dict(attrs)
        if tag == "li" and a.get("value") == FAMILY_ITEM and not self.in_family:
            self.in_family = True
            self.depth = 0
            self.stack = [None]
            return
        if not self.in_family:
            return
        if tag == "ol":
            self.depth += 1
            self.stack.append(None)
        elif tag == "a" and a.get("href", "").startswith("#"):
            self._href = a["href"][1:]
            self._text = []

    def handle_data(self, data):
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag):
        if not self.in_family or self.done:
            return
        if tag == "a" and self._href is not None:
            if self.depth >= 1:
                parent = self.stack[self.depth - 1] if self.depth - 1 >= 1 else None
                self.items.append((self._href, "".join(self._text).strip(), parent))
                self.stack[self.depth] = self._href
            self._href = None
        elif tag == "ol":
            self.stack.pop()
            self.depth -= 1
            if self.depth == 0:
                self.done = True


_SECTION_RE = re.compile(r'<section class="showtarget" id="([^"]+)">(.*?)</section>', re.S)
_H2_RE = re.compile(r"<h2>([\d.]+)\s+(.*?)</h2>", re.S)
_RULE_RE = re.compile(r'<p class="schg"[^>]*>(.*?)(?=<p|</section>|$)', re.S)
_ATTR_RE = re.compile(r"</h2>\s*<p>(.*?)(?=<p|$)", re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def _text(fragment: str) -> str:
    return html.unescape(_TAG_RE.sub("", fragment)).strip()


def _sections(page: str) -> dict[str, tuple[str, str, str, list[str]]]:
    out = {}
    for sid, body in _SECTION_RE.findall(page + "</section>"):
        h2 = _H2_RE.search(body)
        if not h2:
            continue
        attr = _ATTR_RE.search(body)
        rules = [_text(r) for r in _RULE_RE.findall(body)]
        out[sid] = (h2.group(1), _text(h2.group(2)), _text(attr.group(1)) if attr else "", rules)
    return out


def _split_title(title: str) -> tuple[str | None, str | None]:
    if " to " in title:
        src, tgt = title.split(" to ", 1)
        return src.strip(), tgt.strip()
    return None, None


def _bullet_ancestors(nodes: dict[str, Node], nid: str) -> list[str]:
    chain, cur = [], nodes[nid].bullet_parent
    while cur:
        chain.append(cur)
        cur = nodes[cur].bullet_parent
    return chain


def build(page: str) -> dict[str, Node]:
    toc = _TocParser()
    toc.feed(page)
    sections = _sections(page)

    nodes: dict[str, Node] = {}
    for sid, toc_title, bparent in toc.items:
        number, title, attribution, rules = sections.get(sid, ("", toc_title, "", []))
        src, tgt = _split_title(title)
        nodes[sid] = Node(sid, number, title, src, tgt, bparent, attribution=attribution, rules=rules)

    has_children = {n.bullet_parent for n in nodes.values() if n.bullet_parent}
    for n in nodes.values():
        if n.target is None:
            n.kind = "container" if n.id in has_children else "process"

    # Containers pass their bullet role on to their own bullet parent.
    def effective_bullet_parent(nid: str | None) -> str | None:
        while nid and nodes[nid].kind == "container":
            nid = nodes[nid].bullet_parent
        return nid

    for n in nodes.values():
        n.bullet_parent = effective_bullet_parent(n.bullet_parent)

    # Resolve stage parents from titles, nearest in bullet nesting first.
    by_target: dict[str, list[str]] = {}
    for n in nodes.values():
        if n.kind == "stage":
            by_target.setdefault(n.target, []).append(n.id)

    for n in nodes.values():
        if n.kind != "stage":
            continue
        if n.source == ROOT_NAME:
            n.parent = None
            continue
        candidates = [c for c in by_target.get(n.source, []) if c != n.id]
        if candidates:
            scope = [n.id] + _bullet_ancestors(nodes, n.id)

            def closeness(c: str) -> int:
                c_scope = [c] + _bullet_ancestors(nodes, c)
                for i, s in enumerate(scope):
                    if s in c_scope:
                        return i
                return len(scope)

            n.parent = min(candidates, key=closeness)
        else:
            n.parent = n.bullet_parent

    # Process-only sections become dead-end side branches under their bullet
    # parent. A language passes through one only when its config names it.
    for n in nodes.values():
        if n.kind == "process":
            n.parent = n.bullet_parent
    return nodes


def lineage(nodes: dict[str, Node], nid: str) -> list[str]:
    """Stage ids from the given stage up to (not including) the root, youngest first."""
    chain, cur = [], nid
    while cur:
        chain.append(cur)
        cur = nodes[cur].parent
    return chain


def nearest_common_ancestor(nodes: dict[str, Node], a: str, b: str) -> str | None:
    """Stage id of the nearest shared ancestor, or None when only the root is shared."""
    up_a = lineage(nodes, a)
    for s in lineage(nodes, b):
        if s in up_a:
            return s
    return None


def save(nodes: dict[str, Node], path: Path = TREE_JSON) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "source": "Index Diachronica (https://chridd.nfshost.com/diachronica/all), item 17",
        "license": "CC BY-NC-SA 3.0",
        "root": ROOT_NAME,
        "nodes": {k: asdict(v) for k, v in nodes.items()},
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1))


def load(path: Path = TREE_JSON) -> dict[str, Node]:
    data = json.loads(path.read_text())
    return {k: Node(**v) for k, v in data["nodes"].items()}


def main() -> None:
    nodes = build(SOURCE_HTML.read_text())
    save(nodes)
    print(f"Wrote {len(nodes)} nodes to {TREE_JSON}")


if __name__ == "__main__":
    main()
