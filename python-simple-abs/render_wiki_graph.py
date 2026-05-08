"""Render a graph of wiki-page links.

Usage:
  python python/render_wiki_graph.py
  python python/render_wiki_graph.py --open

Outputs (default):
  wiki/wiki-links.dot
  wiki/wiki-links.svg   (if Graphviz `dot` is installed)
  on-screen window      (shown by default)
"""

from __future__ import annotations

import argparse
import math
import re
import subprocess
import colorsys
from pathlib import Path
from html.parser import HTMLParser
from typing import Dict, List, Set, Tuple


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent


def _select_wiki_dir() -> Path:
    candidates = [ROOT / "wiki", REPO_ROOT / "wiki"]
    existing = [p for p in candidates if p.is_dir()]
    if not existing:
        return candidates[0]
    # Prefer the directory that actually contains the wiki corpus.
    return max(existing, key=lambda p: len(list(p.glob("*.html"))))


WIKI_DIR = _select_wiki_dir()
DOT_OUT = WIKI_DIR / "wiki-links.dot"
SVG_OUT = WIKI_DIR / "wiki-links.svg"


class _HrefParser(HTMLParser):
    def __init__(self, keep_nav_links: bool = False) -> None:
        super().__init__()
        self.hrefs: List[str] = []
        self._in_nav_depth = 0
        self._keep_nav_links = keep_nav_links

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, str | None]]) -> None:
        if tag.lower() == "nav":
            self._in_nav_depth += 1
            return
        if self._in_nav_depth > 0 and not self._keep_nav_links:
            return
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "nav" and self._in_nav_depth > 0:
            self._in_nav_depth -= 1


def _extract_links(html_path: Path) -> List[str]:
    # Keep nav links only for index page so hierarchy has a meaningful root.
    keep_nav = html_path.name == "index.html"
    p = _HrefParser(keep_nav_links=keep_nav)
    p.feed(html_path.read_text(encoding="utf-8", errors="ignore"))
    return p.hrefs


def _normalize_target(href: str) -> str | None:
    # Keep only local html links
    if not href or href.startswith(("http://", "https://", "mailto:", "#")):
        return None
    href = href.split("#", 1)[0].strip()
    if not href:
        return None
    # Normalize relative refs like ./foo.html
    href = re.sub(r"^\./", "", href)
    if not href.endswith(".html"):
        return None
    # Keep only page filename in wiki scope
    return Path(href).name


def build_graph(excluded: Set[str]) -> Tuple[Set[str], Set[Tuple[str, str]]]:
    pages = {
        p.name
        for p in WIKI_DIR.glob("*.html")
        if p.is_file() and p.name not in excluded
    }
    edges: Set[Tuple[str, str]] = set()

    for page in sorted(pages):
        src = WIKI_DIR / page
        for href in _extract_links(src):
            dst = _normalize_target(href)
            if not dst:
                continue
            if dst in pages:
                if dst != page:
                    edges.add((page, dst))

    return pages, edges


def write_dot(nodes: Set[str], edges: Set[Tuple[str, str]], dot_path: Path) -> None:
    dot_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "digraph wiki {",
        "  rankdir=LR;",
        '  graph [fontname="Helvetica"];',
        '  node [shape=box, style=rounded, fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]
    for n in sorted(nodes):
        lines.append(f'  "{n}";')
    for a, b in sorted(edges):
        lines.append(f'  "{a}" -> "{b}";')
    lines.append("}")
    dot_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_svg(dot_path: Path, svg_path: Path) -> bool:
    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except Exception:
        return False


def show_graph(nodes: Set[str], edges: Set[Tuple[str, str]]) -> None:
    import tkinter as tk

    if not nodes:
        return

    w, h = 1400, 900
    box_w, box_h = 210, 34
    margin_x, margin_y = 60, 50

    # Build adjacency
    adj: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        adj.setdefault(a, []).append(b)

    # BFS levels from index.html
    root_name = "index.html" if "index.html" in nodes else sorted(nodes)[0]
    level_of: Dict[str, int] = {root_name: 0}
    q: List[str] = [root_name]
    while q:
        cur = q.pop(0)
        for nxt in adj.get(cur, []):
            if nxt not in level_of:
                level_of[nxt] = level_of[cur] + 1
                q.append(nxt)

    # Unreached nodes go to the last level
    max_level = max(level_of.values()) if level_of else 0
    for n in nodes:
        if n not in level_of:
            level_of[n] = max_level + 1
    max_level = max(level_of.values()) if level_of else 0

    levels: Dict[int, List[str]] = {}
    for n, lvl in level_of.items():
        levels.setdefault(lvl, []).append(n)
    for lvl in levels:
        levels[lvl] = sorted(levels[lvl])

    # Optimize node order within each level to keep children close to parents.
    parents_of: Dict[str, List[str]] = {n: [] for n in nodes}
    for a, b in edges:
        parents_of[b].append(a)

    for _ in range(4):
        for lvl in range(1, max_level + 1):
            row = levels.get(lvl, [])
            prev = levels.get(lvl - 1, [])
            if not row or not prev:
                continue
            prev_idx = {name: i for i, name in enumerate(prev)}

            def bary(name: str) -> float:
                ps = [p for p in parents_of.get(name, []) if p in prev_idx]
                if not ps:
                    # Keep currently unanchored nodes roughly stable
                    return float(row.index(name))
                return sum(prev_idx[p] for p in ps) / len(ps)

            levels[lvl] = sorted(row, key=bary)

    # Color assignment:
    # - Level 1 nodes: pastel rainbow
    # - Deeper nodes: blend of parent node colors
    def _rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
        r, g, b = rgb
        return "#{:02x}{:02x}{:02x}".format(
            max(0, min(255, int(round(r * 255)))),
            max(0, min(255, int(round(g * 255)))),
            max(0, min(255, int(round(b * 255)))),
        )

    def _pastel(h: float) -> Tuple[float, float, float]:
        # Pastel tone via high lightness and moderate saturation.
        r, g, b = colorsys.hls_to_rgb(h % 1.0, 0.84, 0.50)
        return (r, g, b)

    node_color: Dict[str, Tuple[float, float, float]] = {}
    node_color[root_name] = (0.92, 0.94, 0.98)
    lvl1 = levels.get(1, [])
    n1 = max(1, len(lvl1))
    for i, name in enumerate(lvl1):
        node_color[name] = _pastel(i / n1)

    for lvl in range(2, max_level + 1):
        for name in levels.get(lvl, []):
            # Blend from already-colored parents at higher levels.
            ps = [p for p in parents_of.get(name, []) if p in node_color and level_of[p] < lvl]
            if ps:
                rr = sum(node_color[p][0] for p in ps) / len(ps)
                gg = sum(node_color[p][1] for p in ps) / len(ps)
                bb = sum(node_color[p][2] for p in ps) / len(ps)
                node_color[name] = (rr, gg, bb)
            else:
                node_color[name] = (0.91, 0.96, 0.94)

    # Fill any unassigned nodes.
    for name in nodes:
        node_color.setdefault(name, (0.91, 0.96, 0.94))

    # Assign positions level-by-level
    usable_w = w - 2 * margin_x
    usable_h = h - 2 * margin_y
    y_step = usable_h / max(1, max_level + 1)
    pos: Dict[str, Tuple[float, float]] = {}
    for lvl in range(max_level + 1):
        row = levels.get(lvl, [])
        if not row:
            continue
        x_step = usable_w / (len(row) + 1)
        y = margin_y + lvl * y_step
        for i, name in enumerate(row, start=1):
            x = margin_x + i * x_step
            pos[name] = (x, y)

    root = tk.Tk()
    root.title("Wiki Link Graph")
    canvas = tk.Canvas(root, width=w, height=h, bg="white")
    canvas.pack(fill="both", expand=True)

    # edges: draw back/up/same-level first (light, thin), then down-hierarchy on top (dark, thick)
    up_or_cross: List[Tuple[str, str]] = []
    down: List[Tuple[str, str]] = []
    for a, b in sorted(edges):
        if level_of[b] > level_of[a]:
            down.append((a, b))
        else:
            up_or_cross.append((a, b))

    for a, b in up_or_cross:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill="#cbd5e1",
            width=1.0,
            arrow=tk.LAST,
            arrowshape=(8, 10, 3),
        )

    for a, b in down:
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        canvas.create_line(
            x1,
            y1,
            x2,
            y2,
            fill="#1f4e78",
            width=2.8,
            arrow=tk.LAST,
            arrowshape=(10, 12, 4),
        )

    # nodes
    theta = math.radians(45.0)
    ct = math.cos(theta)
    st = math.sin(theta)

    for name in sorted(nodes):
        x, y = pos[name]
        # Rotated filled background (no outline) matching text angle.
        # Swap long/short axes for the rotated background shape.
        hw = box_h / 2
        hh = box_w / 2
        base = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        pts: List[float] = []
        for px, py in base:
            rx = px * ct - py * st
            ry = px * st + py * ct
            pts.extend([x + rx, y + ry])
        canvas.create_polygon(*pts, fill=_rgb_to_hex(node_color[name]), outline="#1f4e78", width=1.2)
        canvas.create_text(x, y, text=name, fill="#0f172a", font=("Segoe UI", 9), angle=45)

    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--exclude",
        action="append",
        default=["python-estimates.html"],
        help="Wiki HTML page filename to exclude (repeatable).",
    )
    ap.add_argument(
        "--open",
        action="store_true",
        help="Open rendered SVG if available.",
    )
    ap.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display graph window.",
    )
    args = ap.parse_args()

    excluded = set(args.exclude)
    nodes, edges = build_graph(excluded)
    write_dot(nodes, edges, DOT_OUT)

    rendered = render_svg(DOT_OUT, SVG_OUT)
    print(f"Wrote: {DOT_OUT}")
    if rendered:
        print(f"Wrote: {SVG_OUT}")
        if args.open:
            subprocess.run(["cmd", "/c", "start", "", str(SVG_OUT)], check=False)
    else:
        print("Graphviz `dot` not found; SVG not rendered. Install Graphviz or use the .dot file.")

    if not args.no_show:
        show_graph(nodes, edges)


if __name__ == "__main__":
    main()
