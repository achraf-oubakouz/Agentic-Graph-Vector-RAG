"""
Embeddings Service — Module B

Pipeline:
    TF-IDF (sparse) → LSA/TruncatedSVD (dense, normalised) → PCA 2D

Outputs:
    - lsa matrix, fitted vec/svd/pca objects  (stored in AppStore)
    - cluster labels per chunk                (KMeans on LSA)
    - B1_pca2d_embeddings.png                 (two-panel plot)
        Left  : semantic clusters
        Right : proximity of top-k chunks to query
"""

import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

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
TFIDF_FEATURES = 12_000
LSA_DIMS       = 100
N_CLUSTERS     = 8
TOP_K_PROXIMITY = 10


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, filename: str) -> str:
    path = Path(settings.STATIC_DIR) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    return f"/static/{filename}"


def _setup_ax(ax, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(BG)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#CBCAC4")
    ax.tick_params(colors=MUT)
    ax.grid(True, color=GRID_C, lw=0.5)
    ax.set_xlabel(xlabel, fontsize=8.5, color=MUT)
    ax.set_ylabel(ylabel, fontsize=8.5, color=MUT)


# ══════════════════════════════════════════════════════════════════════════════
#  CORE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def build_embeddings(
    chunks: list[str],
) -> tuple[np.ndarray, np.ndarray, TfidfVectorizer,
           TruncatedSVD, PCA]:
    """
    Fit TF-IDF → LSA → PCA on the given chunks.

    Returns
    -------
    lsa    : (n_chunks, LSA_DIMS)  normalised dense matrix
    coords : (n_chunks, 2)         PCA 2D projection
    vec    : fitted TfidfVectorizer
    svd    : fitted TruncatedSVD
    pca    : fitted PCA
    """
    # TF-IDF
    vec = TfidfVectorizer(
        max_features=TFIDF_FEATURES,
        sublinear_tf=True,
        ngram_range=(1, 2),
    )
    tfidf = vec.fit_transform(chunks)

    # LSA — reduce to dense semantic space
    n_comp = min(LSA_DIMS, len(chunks) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    lsa = normalize(svd.fit_transform(tfidf))

    # PCA 2D for visualisation only
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(lsa)

    return lsa, coords, vec, svd, pca


# ══════════════════════════════════════════════════════════════════════════════
#  PLOT
# ══════════════════════════════════════════════════════════════════════════════

def _plot_embeddings(
    coords: np.ndarray,
    labels: np.ndarray,
    top_k: np.ndarray,
    q_xy: np.ndarray,
    var: np.ndarray,
    n_clusters: int,
    query: str,
) -> str:
    """B1 — two-panel PCA scatter plot."""
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    fig.suptitle(
        f"Module B — PCA 2D  |  {len(coords)} chunks  |  "
        f"PC1={var[0]:.1%}  PC2={var[1]:.1%}",
        fontsize=11, fontweight="bold", color=TXT,
    )

    pc1_lbl = f"PC1 ({var[0]:.1%})"
    pc2_lbl = f"PC2 ({var[1]:.1%})"

    # ── Left panel : semantic clusters ───────────────────────────────────────
    _setup_ax(ax_l, pc1_lbl, pc2_lbl)
    cluster_colors = [PALETTE[l % len(PALETTE)] for l in labels]
    ax_l.scatter(coords[:, 0], coords[:, 1],
                 c=cluster_colors, alpha=0.6, s=16)

    for cl in range(n_clusters):
        mask = labels == cl
        if mask.sum() == 0:
            continue
        cx = coords[mask, 0].mean()
        cy = coords[mask, 1].mean()
        ax_l.scatter(cx, cy, s=100,
                     c=PALETTE[cl % len(PALETTE)],
                     edgecolors="white", marker="D", zorder=5)
        ax_l.text(cx, cy + 0.02, f"C{cl + 1}",
                  fontsize=7, ha="center", weight="bold", color=TXT)

    ax_l.set_title(f"Clusters sémantiques ({n_clusters} groupes)",
                   fontsize=10, color=TXT)

    # ── Right panel : proximity to query ─────────────────────────────────────
    _setup_ax(ax_r, pc1_lbl, pc2_lbl)
    ax_r.scatter(coords[:, 0], coords[:, 1],
                 c="#CCCCCC", alpha=0.3, s=10)
    ax_r.scatter(coords[top_k, 0], coords[top_k, 1],
                 c=PALETTE[2], alpha=0.9, s=55,
                 edgecolors="white", zorder=5,
                 label=f"Top-{TOP_K_PROXIMITY}")

    for rank, idx in enumerate(top_k):
        ax_r.text(coords[idx, 0] + 0.01, coords[idx, 1],
                  f"#{rank + 1}", fontsize=7,
                  color="#A03535", weight="bold")

    # Query star
    ax_r.scatter([q_xy[0]], [q_xy[1]],
                 c="#EDC948", s=200, edgecolors=TXT,
                 marker="*", zorder=6, label="Requête")

    # Proximity circle
    dists = np.sqrt((coords[:, 0] - q_xy[0])**2
                    + (coords[:, 1] - q_xy[1])**2)
    radius = float(np.sort(dists)[TOP_K_PROXIMITY])
    circle = plt.Circle((q_xy[0], q_xy[1]), radius,
                         fill=False, ls="--",
                         color="#EDC948", lw=1.2, alpha=0.7)
    ax_r.add_patch(circle)

    short_query = query[:50] + "…" if len(query) > 50 else query
    ax_r.set_title(
        f"Proximité — Top-{TOP_K_PROXIMITY} chunks\nRequête : {short_query}",
        fontsize=10, color=TXT,
    )
    ax_r.legend(fontsize=8)

    plt.tight_layout()
    return _save(fig, "B1_pca2d_embeddings.png")


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_embeddings(
    chunks: list[str],
    query: str,
    lsa: np.ndarray,
    vec: TfidfVectorizer,
    svd: TruncatedSVD,
    pca: PCA,
) -> dict:
    """
    Project chunks into 2D PCA space, cluster them, find top-k
    closest to the query, and generate the visualisation.

    The caller (router) passes the already-fitted objects from
    the store so we never refit unnecessarily.

    Returns a dict ready to be unpacked into EmbeddingsResponse.
    """
    var    = pca.explained_variance_ratio_
    coords = pca.transform(lsa)

    # KMeans clustering on the full LSA space (not PCA)
    n_cl = max(2, min(N_CLUSTERS, len(chunks) // 5, len(chunks) - 1))
    km   = KMeans(n_clusters=n_cl, random_state=42, n_init=10)
    labels = km.fit_predict(lsa)

    # Project query into LSA → PCA space
    q_lsa = normalize(svd.transform(vec.transform([query])))
    q_xy  = pca.transform(q_lsa)[0]

    # Cosine similarity between query and all chunks
    sims  = cosine_similarity(q_lsa, lsa)[0]
    top_k = np.argsort(sims)[::-1][:TOP_K_PROXIMITY]
    cluster_sizes = np.bincount(labels, minlength=n_cl)
    metrics = {
        "tfidf_features": int(len(vec.get_feature_names_out())),
        "lsa_dimensions": int(lsa.shape[1]),
        "pca_components": 2,
        "pca_total_variance": float(var[:2].sum()),
        "silhouette_score": None,
        "davies_bouldin_score": None,
        "calinski_harabasz_score": None,
        "top_k_similarity_mean": float(np.mean(sims[top_k])),
        "top_k_similarity_min": float(np.min(sims[top_k])),
        "top_k_similarity_max": float(np.max(sims[top_k])),
        "query_vector_norm": float(np.linalg.norm(q_lsa)),
        "cluster_size_min": int(cluster_sizes.min()),
        "cluster_size_max": int(cluster_sizes.max()),
        "cluster_size_mean": float(cluster_sizes.mean()),
    }
    if 1 < n_cl < len(chunks):
        metrics["silhouette_score"] = float(silhouette_score(lsa, labels, metric="cosine"))
        metrics["davies_bouldin_score"] = float(davies_bouldin_score(lsa, labels))
        metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(lsa, labels))

    img = _plot_embeddings(coords, labels, top_k, q_xy, var, n_cl, query)

    return {
        "pca_coords":         coords.tolist(),
        "cluster_labels":     labels.tolist(),
        "explained_variance": var.tolist(),
        "top_k_indices":      top_k.tolist(),
        "query_point":        q_xy.tolist(),
        "image_url":          img,
        "metrics":            metrics,
    }
