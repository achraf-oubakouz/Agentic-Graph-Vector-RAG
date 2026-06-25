from pydantic import BaseModel
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
#  CORPUS
# ══════════════════════════════════════════════════════════════════════════════

class CorpusUploadResponse(BaseModel):
    status: str
    filename: str
    size_chars: int
    size_words: int


# ══════════════════════════════════════════════════════════════════════════════
#  CHUNKING
# ══════════════════════════════════════════════════════════════════════════════

class ChunkMetrics(BaseModel):
    n_chunks: int
    avg_len: float
    std_len: float
    vocab_density: float
    score: float


class ChunkingResponse(BaseModel):
    best_method: str
    num_chunks: int
    # Metrics for all 7 methods so the frontend can display comparisons
    all_metrics: dict[str, ChunkMetrics]
    comparison_image: str   # URL path → /static/A1_chunking_comparison.png
    hierarchy_image: str    # URL path → /static/A2_hierarchy_tree.png


# ══════════════════════════════════════════════════════════════════════════════
#  EMBEDDINGS
# ══════════════════════════════════════════════════════════════════════════════

class EmbeddingsRequest(BaseModel):
    query: str


class EmbeddingsResponse(BaseModel):
    pca_coords: list[list[float]]       # [[x, y], ...] one per chunk
    cluster_labels: list[int]           # cluster id per chunk
    explained_variance: list[float]     # [PC1_ratio, PC2_ratio]
    top_k_indices: list[int]            # indices of closest chunks to query
    query_point: list[float]            # [x, y] of query in PCA space
    image_url: str                      # /static/B1_pca2d_embeddings.png
    metrics: dict[str, Any] = {}        # embedding quality and similarity metrics


# ══════════════════════════════════════════════════════════════════════════════
#  RETRIEVAL
# ══════════════════════════════════════════════════════════════════════════════

class RetrievalRequest(BaseModel):
    query: str


class RetrievalMethodScore(BaseModel):
    method: str
    avg_score: float
    diversity: float
    harmonic: float


class RetrievedChunk(BaseModel):
    rank: int
    score: float
    text: str


class RetrievalResponse(BaseModel):
    best_method: str
    comparison: list[RetrievalMethodScore]   # scores for all 5 methods
    top_results: list[RetrievedChunk]         # top-k chunks from best method
    resume: str                               # synthesised summary
    image_url: str                            # /static/C1_retrieval_comparison.png
    vector_store_path: str                    # where the FAISS index was saved


# ══════════════════════════════════════════════════════════════════════════════
#  QUERY ROUTER
# ══════════════════════════════════════════════════════════════════════════════

class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    query: str
    detected_type: str    # "semantic" | "systematic" | "hybrid"
    routing: str          # "Vectorial RAG" | "Graph RAG" | ...
    confidence: float


# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY
# ══════════════════════════════════════════════════════════════════════════════

class HistoryResponse(BaseModel):
    queries: list[QueryResponse]


class GraphSearchRequest(BaseModel):
    query: str
    limit: int = 25


class GraphPathResult(BaseModel):
    source: str
    relationship: str
    target: str
    source_community: int | None = None
    target_community: int | None = None
    weight: float | None = None


class GraphSearchResponse(BaseModel):
    query: str
    results: list[GraphPathResult]
    communities: list[dict[str, Any]]


class AgenticAskRequest(BaseModel):
    query: str
    limit: int = 8


class AgenticAskResponse(BaseModel):
    query: str
    route: str
    detected_type: str = ""
    policy_scores: dict[str, float] = {}
    llm_model: str = ""
    llm_synthesis: bool = False
    cached: bool = False
    answer: str
    vector_results: list[RetrievedChunk] = []
    graph_results: list[GraphPathResult] = []
    communities: list[dict[str, Any]] = []


class AgenticFeedbackRequest(BaseModel):
    query: str
    route: str
    reward: float


class AgenticFeedbackResponse(BaseModel):
    state: str
    next_state: str = ""
    route: str
    reward: float
    alpha: float = 0.0
    gamma: float = 0.0
    td_target: float = 0.0
    td_error: float = 0.0
    updated_q_value: float
    policy_scores: dict[str, float]
