"""
Chunking Service — Module A

7 chunking methods:
    1. Fixed-size
    2. Sentences
    3. Paragraphs
    4. Semantic similarity
    5. Sliding window
    6. Recursive
    7. Topic structure

Outputs:
    - best method name + its chunks
    - metrics for all 7 methods
    - A1_chunking_comparison.png  (bar chart)
    - A2_hierarchy_tree.png       (Document → Chunks → Sub-Chunks)
"""

import re
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.core.config import settings

warnings.filterwarnings("ignore")

# ── Visual constants ──────────────────────────────────────────────────────────
PALETTE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
           "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]
BG     = "#F7F7F5"
GRID_C = "#E0DDD8"
TXT    = "#2C2C2A"
MUT    = "#7A7870"

# ── Chunking parameters ───────────────────────────────────────────────────────
FIXED_CHARS   = 500
WINDOW_CHARS  = 600
OVERLAP_CHARS = 100
SEM_THRESHOLD = 0.25
RECURSIVE_MAX = 500
SUBCHUNK_DIV  = 3


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _tokens(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _ax_style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBCAC4")
    ax.tick_params(colors=MUT, labelsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold", color=TXT, pad=7)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=MUT)
    ax.yaxis.grid(True, color=GRID_C, lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def _save(fig, filename: str) -> str:
    """Save figure to static/ and return the URL path."""
    path = Path(settings.STATIC_DIR) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return f"/static/{filename}"


# ══════════════════════════════════════════════════════════════════════════════
#  7 CHUNKING METHODS
# ══════════════════════════════════════════════════════════════════════════════

def _split_sentences(text: str) -> list[str]:
    pattern = r'(?<=[.!?])\s+(?=[A-ZÁÀÂÉÈÊÎÏÔÙÛÜŸ"\(])'
    return [s.strip() for s in re.split(pattern, text) if len(s.strip()) > 20]


def chunk_fixed_size(text: str) -> list[str]:
    return [
        text[i:i + FIXED_CHARS].strip()
        for i in range(0, len(text), FIXED_CHARS)
        if len(text[i:i + FIXED_CHARS].strip()) > 30
    ]


def chunk_sentences(text: str) -> list[str]:
    sentences = _split_sentences(text)
    chunks, buf = [], ""
    for s in sentences:
        if len(buf) + len(s) < 600:
            buf = f"{buf} {s}".strip()
        else:
            if buf:
                chunks.append(buf)
            buf = s
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) > 30]


def chunk_paragraphs(text: str) -> list[str]:
    return [
        p.strip()
        for p in re.split(r'\n\s*\n', text)
        if len(p.strip()) > 50
    ]


def chunk_semantic(text: str) -> list[str]:
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return chunk_paragraphs(text)
    vec = TfidfVectorizer(max_features=6000, sublinear_tf=True)
    mat = vec.fit_transform(sentences).toarray()
    chunks, cur_s, cur_i = [], [sentences[0]], [0]
    for i in range(1, len(sentences)):
        centroid = mat[cur_i].mean(axis=0, keepdims=True)
        sim = cosine_similarity(centroid, mat[i:i + 1])[0, 0]
        if sim >= SEM_THRESHOLD:
            cur_s.append(sentences[i])
            cur_i.append(i)
        else:
            chunks.append(" ".join(cur_s))
            cur_s, cur_i = [sentences[i]], [i]
    if cur_s:
        chunks.append(" ".join(cur_s))
    return [c for c in chunks if len(c) > 30]


def chunk_sliding_window(text: str) -> list[str]:
    step = WINDOW_CHARS - OVERLAP_CHARS
    return [
        text[i:i + WINDOW_CHARS].strip()
        for i in range(0, len(text), step)
        if len(text[i:i + WINDOW_CHARS].strip()) > 50
    ]


def chunk_recursive(text: str) -> list[str]:
    SEPS = ["\n\n", "\n", ". ", " "]

    def _split(t: str, seps: list) -> list[str]:
        if len(t) <= RECURSIVE_MAX or not seps:
            return [t.strip()] if t.strip() else []
        parts = t.split(seps[0])
        out = []
        for p in parts:
            if len(p) <= RECURSIVE_MAX:
                if p.strip():
                    out.append(p.strip())
            else:
                out.extend(_split(p, seps[1:]))
        return out

    pieces = _split(text, SEPS)
    merged, buf = [], ""
    for p in pieces:
        if len(buf) + len(p) < RECURSIVE_MAX:
            buf = f"{buf} {p}".strip()
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        merged.append(buf)
    return [c for c in merged if len(c) > 30]


def chunk_topic_structure(text: str) -> list[str]:
    pat = re.compile(
        r'(?m)^(?:\d+(?:\.\d+)*[\s.\-]+\w'
        r'|[A-ZÀÂÉÈÊÎÏÔÙÛÜŸÆŒ][A-ZÀÂÉÈÊÎÏÔÙÛÜŸÆŒ ]{3,}'
        r'|---+)'
    )
    positions = [m.start() for m in pat.finditer(text)]
    if not positions:
        return chunk_paragraphs(text)
    positions = [0] + positions + [len(text)]
    return [
        text[positions[i]:positions[i + 1]].strip()
        for i in range(len(positions) - 1)
        if len(text[positions[i]:positions[i + 1]].strip()) > 50
    ]


# Registry — order matters for display
CHUNKING_METHODS: dict[str, callable] = {
    "1.Fixed-size":      chunk_fixed_size,
    "2.Sentences":       chunk_sentences,
    "3.Paragraphs":      chunk_paragraphs,
    "4.Semantic":        chunk_semantic,
    "5.Sliding-Window":  chunk_sliding_window,
    "6.Recursive":       chunk_recursive,
    "7.Topic-Structure": chunk_topic_structure,
}


# ══════════════════════════════════════════════════════════════════════════════
#  SCORING
# ══════════════════════════════════════════════════════════════════════════════

def _eval_chunks(chunks: list[str]) -> dict:
    if not chunks:
        return dict(n_chunks=0, avg_len=0.0, std_len=0.0,
                    vocab_density=0.0, score=0.0)
    lens    = np.array([len(c) for c in chunks])
    all_tok = _tokens(" ".join(chunks))
    density  = len(set(all_tok)) / max(len(all_tok), 1)
    std_norm = float(np.std(lens)) / (float(np.mean(lens)) + 1e-9)
    score = (
        0.5 * density
        + 0.3 * (1 - min(std_norm, 1))
        + 0.2 * min(1, len(chunks) / 200)
    )
    return dict(
        n_chunks=len(chunks),
        avg_len=round(float(np.mean(lens)), 1),
        std_len=round(float(np.std(lens)), 1),
        vocab_density=round(density, 4),
        score=round(score, 4),
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def _plot_comparison(results: dict, best_name: str) -> str:
    """A1 — bar chart comparing all 7 methods on 3 metrics."""
    names    = list(results.keys())
    labels   = [re.sub(r'^\d+\.', '', n) for n in names]
    scores   = [results[n]["metrics"]["score"]    for n in names]
    avglens  = [results[n]["metrics"]["avg_len"]  for n in names]
    nchunks  = [results[n]["metrics"]["n_chunks"] for n in names]
    best_idx = int(np.argmax(scores))
    colors   = [PALETTE[2] if i == best_idx else "#AAAAAA"
                for i in range(len(names))]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.suptitle(
        "Module A — Comparaison des 7 méthodes de Chunking",
        fontsize=13, fontweight="bold", color=TXT, y=1.02,
    )

    for ax, vals, title, ylabel, fmt in [
        (axes[0], scores,  "Score Composite ↑",       "Score", "{:.3f}"),
        (axes[1], avglens, "Longueur Moyenne (chars)", "Chars", "{:.0f}"),
        (axes[2], nchunks, "Nombre de Chunks",         "Count", "{:.0f}"),
    ]:
        bars = ax.bar(range(len(names)), vals,
                      color=colors, alpha=0.85, width=0.6, zorder=3)
        bars[best_idx].set_edgecolor(TXT)
        bars[best_idx].set_linewidth(2)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(labels, fontsize=7.5, rotation=28,
                           ha="right", color=TXT)
        ax.set_ylim(0, max(vals) * 1.28 if max(vals) > 0 else 1)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + max(vals) * 0.015,
                fmt.format(v),
                ha="center", va="bottom", fontsize=7.5, color=TXT,
            )
        _ax_style(ax, title, ylabel)

    axes[0].legend(
        handles=[
            mpatches.Patch(fc=PALETTE[2], label=f"★ {names[best_idx]}"),
            mpatches.Patch(fc="#AAAAAA",  label="Autres méthodes"),
        ],
        fontsize=8, loc="upper right", framealpha=0.9,
    )
    plt.tight_layout()
    return _save(fig, "A1_chunking_comparison.png")


def _plot_hierarchy(text: str, method_name: str, chunks: list[str]) -> str:
    """A2 — Document → Chunks → Sub-Chunks tree."""

    def make_subs(chunk: str) -> list[str]:
        size = max(60, len(chunk) // SUBCHUNK_DIV)
        return [
            chunk[i:i + size].strip()
            for i in range(0, len(chunk), size)
            if chunk[i:i + size].strip()
        ]

    sub_chunks = [make_subs(c) for c in chunks]
    SHOW = min(7, len(chunks))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax.axis("off")
    ax.set_title(
        f"Hiérarchie — {method_name}  |  {len(chunks)} Chunks",
        fontsize=11, fontweight="bold", color=TXT,
    )
    Y_DOC, Y_CHUNK, Y_SUB = 0.88, 0.58, 0.22

    def _node(x, y, txt, fc, ec, fs=8.5):
        ax.text(
            x, y, txt,
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=fs, color="white",
            bbox=dict(boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=1.3),
        )

    def _arrow(xy, xytext, color):
        ax.annotate(
            "", xy=xy, xytext=xytext,
            xycoords="axes fraction", textcoords="axes fraction",
            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                            connectionstyle="arc3,rad=0"),
        )

    _node(0.5, Y_DOC,
          f"Document Complet\n{len(text):,} chars",
          "#2C6FAC", "#1A4F7A", fs=9)

    xs = np.linspace(0.06, 0.94, SHOW)
    for ci, (cx, chunk) in enumerate(zip(xs, chunks[:SHOW])):
        _arrow((cx, Y_CHUNK + 0.07), (0.5, Y_DOC - 0.05), "#4E79A7")
        _node(cx, Y_CHUNK,
              f"Chunk {ci + 1}\n{len(chunk)} ch",
              "#59A14F", "#3A7035", fs=7)
        subs = sub_chunks[ci][:3]
        if not subs:
            continue
        sxs = np.linspace(
            cx - 0.03 * (len(subs) - 1),
            cx + 0.03 * (len(subs) - 1),
            len(subs),
        )
        for si, (sx, sub) in enumerate(zip(sxs, subs)):
            _arrow((sx, Y_SUB + 0.06), (cx, Y_CHUNK - 0.05), "#F28E2B")
            _node(sx, Y_SUB,
                  f"S{ci + 1}.{si + 1}\n{len(sub)}c",
                  "#E15759", "#A03535", fs=6.5)

    ax.legend(
        handles=[
            mpatches.Patch(fc="#2C6FAC", label="Document"),
            mpatches.Patch(fc="#59A14F", label="Chunks"),
            mpatches.Patch(fc="#E15759", label="Sub-Chunks"),
        ],
        fontsize=8.5, loc="lower right",
    )
    plt.tight_layout()
    return _save(fig, "A2_hierarchy_tree.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_chunking(text: str) -> dict:
    """
    Run all 7 methods, pick the best, generate both plots.

    Returns a dict ready to be unpacked into ChunkingResponse.
    """
    results = {}
    for name, fn in CHUNKING_METHODS.items():
        chunks = fn(text)
        results[name] = {
            "chunks":  chunks,
            "metrics": _eval_chunks(chunks),
        }

    best_name   = max(results, key=lambda k: results[k]["metrics"]["score"])
    best_chunks = results[best_name]["chunks"]

    img_comparison = _plot_comparison(results, best_name)
    img_hierarchy  = _plot_hierarchy(text, best_name, best_chunks)

    return {
        "best_method":      best_name,
        "best_chunks":      best_chunks,
        "num_chunks":       results[best_name]["metrics"]["n_chunks"],
        "all_metrics":      {n: results[n]["metrics"] for n in results},
        "comparison_image": img_comparison,
        "hierarchy_image":  img_hierarchy,
    }