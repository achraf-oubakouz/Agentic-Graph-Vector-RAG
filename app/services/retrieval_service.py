"""
Retrieval Service — Module C

5 retrieval methods:
    1. Top-k Semantic      (cosine on LSA embeddings)
    2. Cosine TF-IDF       (cosine on raw TF-IDF vectors)
    3. BM25                (probabilistic term-frequency ranking)
    4. Hybrid BM25 + Emb   (linear combination of 1 and 3)
    5. MMR                 (Maximal Marginal Relevance — diversity)

Each method returns the top-k (chunk_index, score, text) triples.

Evaluation metrics per method:
    - avg_score  : mean relevance score
    - diversity  : 1 − mean pairwise cosine sim among results
    - harmonic   : harmonic mean of avg_score and diversity

Outputs:
    - C1_retrieval_comparison.png   (bar chart — all 5 methods)
    - synthesised résumé            (extractive summary from best results)
    - vector_store.faiss            (FAISS index for semantic retrieval)
    - vector_store_metadata.json    (chunk text + metadata for the FAISS ids)
"""

import re
import math
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.core.config import settings

warnings.filterwarnings("ignore")

# ── Visual constants ──────────────────────────────────────────────────────────
PALETTE = ["#4E79A7", "#F28E2B", "#E15759", "#76B7B2",
           "#59A14F", "#EDC948", "#B07AA1", "#FF9DA7"]
BG    = "#F7F7F5"
GRID_C = "#E0DDD8"
TXT   = "#2C2C2A"
MUT   = "#7A7870"

# ── Parameters ────────────────────────────────────────────────────────────────
TOP_K_RETRIEVAL = 5
MMR_LAMBDA      = 0.6
BM25_K1         = 1.5
BM25_B          = 0.75
HYBRID_ALPHA    = 0.5   # weight for semantic score (1-HYBRID_ALPHA for BM25)


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _tokens(text: str) -> list[str]:
    return re.findall(r'\b\w+\b', text.lower())


def _norm01(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


def _as_faiss_matrix(vectors: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(vectors.astype("float32"))


def _build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    matrix = _as_faiss_matrix(vectors)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def _save(fig, filename: str) -> str:
    path = Path(settings.STATIC_DIR) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return f"/static/{filename}"


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


# ══════════════════════════════════════════════════════════════════════════════
#  BM25
# ══════════════════════════════════════════════════════════════════════════════

def _bm25_scores(query: str, chunks: list[str]) -> np.ndarray:
    tok_docs = [_tokens(c) for c in chunks]
    tok_q    = _tokens(query)
    N        = len(chunks)
    avg_dl   = float(np.mean([len(d) for d in tok_docs]))
    scores   = np.zeros(N)

    for term in tok_q:
        df = sum(1 for d in tok_docs if term in d)
        if not df:
            continue
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        for i, doc in enumerate(tok_docs):
            tf = doc.count(term)
            dl = len(doc)
            num = tf * (BM25_K1 + 1)
            den = tf + BM25_K1 * (1 - BM25_B + BM25_B * dl / avg_dl)
            scores[i] += idf * num / den

    return scores


# ══════════════════════════════════════════════════════════════════════════════
#  5 RETRIEVAL METHODS
# ══════════════════════════════════════════════════════════════════════════════

def _ret_topk_semantic(
    query: str, chunks: list[str],
    lsa: np.ndarray, vec: TfidfVectorizer, svd: TruncatedSVD,
) -> list[tuple[int, float, str]]:
    q = _as_faiss_matrix(normalize(svd.transform(vec.transform([query]))))
    index = _build_faiss_index(lsa)
    scores, ids = index.search(q, min(TOP_K_RETRIEVAL, len(chunks)))
    return [
        (int(i), float(score), chunks[int(i)])
        for score, i in zip(scores[0], ids[0])
        if i >= 0
    ]


def _ret_cosine_tfidf(
    query: str, chunks: list[str],
    vec: TfidfVectorizer, **_,
) -> list[tuple[int, float, str]]:
    q   = vec.transform([query])
    mat = vec.transform(chunks)
    s   = cosine_similarity(q, mat)[0]
    idx = np.argsort(s)[::-1][:TOP_K_RETRIEVAL]
    return [(int(i), float(s[i]), chunks[i]) for i in idx]


def _ret_bm25(
    query: str, chunks: list[str], **_,
) -> list[tuple[int, float, str]]:
    s   = _bm25_scores(query, chunks)
    sn  = _norm01(s)
    idx = np.argsort(sn)[::-1][:TOP_K_RETRIEVAL]
    return [(int(i), float(sn[i]), chunks[i]) for i in idx]


def _ret_hybrid(
    query: str, chunks: list[str],
    lsa: np.ndarray, vec: TfidfVectorizer, svd: TruncatedSVD,
) -> list[tuple[int, float, str]]:
    q   = normalize(svd.transform(vec.transform([query])))
    sem = cosine_similarity(q, lsa)[0]
    bm  = _bm25_scores(query, chunks)
    s   = HYBRID_ALPHA * _norm01(sem) + (1 - HYBRID_ALPHA) * _norm01(bm)
    idx = np.argsort(s)[::-1][:TOP_K_RETRIEVAL]
    return [(int(i), float(s[i]), chunks[i]) for i in idx]


def _ret_mmr(
    query: str, chunks: list[str],
    lsa: np.ndarray, vec: TfidfVectorizer, svd: TruncatedSVD,
) -> list[tuple[int, float, str]]:
    q   = normalize(svd.transform(vec.transform([query])))
    rel = cosine_similarity(q, lsa)[0]
    sel, cands = [], list(range(len(chunks)))

    while len(sel) < TOP_K_RETRIEVAL and cands:
        if not sel:
            best = max(cands, key=lambda i: rel[i])
        else:
            sel_emb = lsa[sel]
            best = max(
                cands,
                key=lambda i: (
                    MMR_LAMBDA * rel[i]
                    - (1 - MMR_LAMBDA)
                    * cosine_similarity(lsa[i:i + 1], sel_emb)[0].max()
                ),
            )
        sel.append(best)
        cands.remove(best)

    return [(int(i), float(rel[i]), chunks[i]) for i in sel]


# ══════════════════════════════════════════════════════════════════════════════
#  EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def _eval_retrieval(
    results: dict[str, list],
    lsa: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for name, res in results.items():
        idx    = [r[0] for r in res]
        scores = np.array([r[1] for r in res])
        if len(idx) > 1:
            sub = lsa[idx]
            sim = cosine_similarity(sub)
            n   = len(sub)
            div = float(1 - sim[np.triu_indices(n, k=1)].mean())
        else:
            div = 0.0
        avg_s    = float(scores.mean())
        harmonic = 2 * div * avg_s / (div + avg_s + 1e-9)
        rows.append(dict(
            method=name,
            avg_score=round(avg_s, 4),
            diversity=round(div, 4),
            harmonic=round(harmonic, 4),
        ))
    return pd.DataFrame(rows).sort_values("harmonic", ascending=False)


# ══════════════════════════════════════════════════════════════════════════════
#  RÉSUMÉ
# ══════════════════════════════════════════════════════════════════════════════

def _make_resume(
    query: str,
    top_results: list[tuple[int, float, str]],
    vec: TfidfVectorizer,
) -> str:
    """Extractive summary — pick the sentences most similar to the query."""
    all_chunks = [c for _, _, c in top_results]
    sents = [
        s.strip()
        for c in all_chunks
        for s in re.split(r'(?<=[.!?])\s+', c)
        if len(s.strip()) > 40
    ]
    if not sents:
        return " ".join(all_chunks)[:800]
    try:
        mat  = vec.transform(sents + [query])
        sims = cosine_similarity(mat[-1:], mat[:-1])[0]
        top  = sorted(np.argsort(sims)[::-1][:6].tolist())
        return " ".join(sents[i] for i in top)
    except Exception:
        return " ".join(sents[:5])


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _plot_comparison(eval_df: pd.DataFrame, best_method: str) -> str:
    """C1 — grouped bar chart comparing all 5 retrieval methods."""
    methods  = eval_df["method"].tolist()
    labels   = [re.sub(r'^\d+\.', '', m) for m in methods]
    x        = np.arange(len(methods))
    w        = 0.26
    best_i   = methods.index(best_method)

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=BG)

    for i, (col, color, lbl) in enumerate([
        ("avg_score", PALETTE[0], "Pertinence"),
        ("diversity", PALETTE[1], "Diversité"),
        ("harmonic",  PALETTE[2], "Harmonique F"),
    ]):
        vals = eval_df[col].tolist()
        bars = ax.bar(x + (i - 1) * w, vals, w,
                      color=color, alpha=0.85, label=lbl)
        if col == "harmonic":
            bars[best_i].set_edgecolor(TXT)
            bars[best_i].set_linewidth(2)
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.015,
                f"{v:.3f}",
                ha="center", fontsize=7, color=TXT,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, color=TXT, fontsize=9)
    ax.set_ylim(0, 1.35)
    ax.legend(fontsize=8)
    _ax_style(
        ax,
        f"Module C — Comparaison des 5 méthodes de Retrieval  "
        f"★ {best_method}",
        "Score",
    )
    plt.tight_layout()
    return _save(fig, "C1_retrieval_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
#  FAISS VECTOR STORE EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def _save_vector_store(
    chunks: list[str],
    lsa: np.ndarray,
    best_method: str,
) -> str:
    """Save the vector store as a FAISS index plus chunk metadata."""
    index_path = Path(settings.FAISS_INDEX_PATH)
    meta_path = Path(settings.FAISS_METADATA_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.parent.mkdir(parents=True, exist_ok=True)

    index = _build_faiss_index(lsa)
    faiss.write_index(index, str(index_path))

    store_data = {
        "metadata": {
            "n_chunks": len(chunks),
            "embedding_dim": int(lsa.shape[1]),
            "best_chunking_method": best_method,
            "index_type": "IndexFlatIP",
            "similarity": "cosine",
            "index_path": str(index_path),
        },
        "chunks": [
            {
                "id": i,
                "text": chunk,
            }
            for i, chunk in enumerate(chunks)
        ],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(store_data, f, ensure_ascii=False, indent=2)

    print(
        f"[retrieval] FAISS vector store saved -> {index_path} "
        f"({len(chunks)} chunks, dim={lsa.shape[1]})"
    )
    print(f"[retrieval] Metadata saved -> {meta_path}")
    return str(index_path)


# ============================================================================
#  PUBLIC ENTRY POINT
# ============================================================================
def run_retrieval(
    query: str,
    chunks: list[str],
    lsa: np.ndarray,
    vec: TfidfVectorizer,
    svd: TruncatedSVD,
    best_chunking_method: str,
    preferred_method: str | None = None,
    result_text_chars: int = 300,
) -> dict:
    """
    Run all 5 retrieval methods, pick the best, generate plot,
    build résumé, and save the FAISS vector store.

    Returns a dict ready to be unpacked into RetrievalResponse.
    """
    # ── Run all 5 methods ────────────────────────────────────────────────────
    raw_results: dict[str, list] = {
        "1.Top-k Semantic":  _ret_topk_semantic(query, chunks, lsa, vec, svd),
        "2.Cosine TF-IDF":   _ret_cosine_tfidf(query, chunks, vec=vec),
        "3.BM25":            _ret_bm25(query, chunks),
        "4.Hybrid BM25+Emb": _ret_hybrid(query, chunks, lsa, vec, svd),
        "5.MMR":             _ret_mmr(query, chunks, lsa, vec, svd),
    }

    # ── Evaluate and pick best ───────────────────────────────────────────────
    eval_df     = _eval_retrieval(raw_results, lsa)
    metric_best_method = eval_df.iloc[0]["method"]
    selected_method = preferred_method if preferred_method in raw_results else metric_best_method
    best_res = raw_results[selected_method]

    # ── Résumé ───────────────────────────────────────────────────────────────
    resume = _make_resume(query, best_res, vec)

    # ── Plot ─────────────────────────────────────────────────────────────────
    img = _plot_comparison(eval_df, metric_best_method)

    # ── Save vector store ────────────────────────────────────────────────────
    vs_path = _save_vector_store(chunks, lsa, best_chunking_method)

    return {
        "best_method":       selected_method,
        "comparison":        eval_df.to_dict(orient="records"),
        "top_results": [
            {"rank": r + 1, "score": score, "text": chunk[:result_text_chars]}
            for r, (_, score, chunk) in enumerate(best_res)
        ],
        "resume":            resume,
        "image_url":         img,
        "vector_store_path": vs_path,
    }

