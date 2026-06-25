# Backend RAG Pipeline Explained

This document explains how the backend of the project works, with emphasis on the RAG pipeline, graph layer, agent orchestration, algorithms, and metrics.

The project is an Agentic Graph-Vector RAG system for a scientific corpus about tropical Pacific warming, CMIP models, SST gradients, model biases, ocean-atmosphere dynamics, wind stress, heat fluxes, Walker circulation, and related mechanisms.

The backend is built with FastAPI and is organized around these main stages:

```text
Corpus text
  -> Chunking
  -> TF-IDF / LSA embeddings
  -> Retrieval comparison + FAISS vector store
  -> Neo4j graph search
  -> Query classification
  -> Q-learning route selection
  -> Vector evidence and/or graph evidence
  -> LLM final synthesis
  -> Optional feedback update
```

## Main Backend Files

The backend entry point is `main.py`. It loads the corpus at startup, registers routers, and exposes the FastAPI app.

Important backend modules:

- `app/core/config.py`: central settings, paths, model configuration, Neo4j credentials, Gemini settings.
- `app/core/database.py`: in-memory app store for corpus text, chunks, embeddings, fitted models, and query history.
- `app/services/chunking_service.py`: chunking algorithms and chunking metrics.
- `app/services/embeddings_service.py`: TF-IDF, TruncatedSVD/LSA, normalization, PCA, KMeans.
- `app/services/retrieval_service.py`: semantic retrieval, TF-IDF retrieval, BM25, hybrid retrieval, MMR, FAISS export, retrieval metrics.
- `app/services/graph_service.py`: Neo4j Aura graph status, graph search, graph network export.
- `app/services/agentic_service.py`: validation, route orchestration, vector/graph evidence retrieval, cache, final response call.
- `app/services/llm_service.py`: evidence formatting, prompt construction, Gemini/local generation, output validation.
- `app/services/rl_service.py`: Q-learning policy storage, route choice, feedback update.
- `app/routers/query.py`: rule-based query classification.

## 1. Corpus Loading

The corpus file is configured by:

```python
CORPUS_PATH = "data/data_final_cleaned.txt"
```

On FastAPI startup, the app loads the corpus text into the shared in-memory store. The store keeps:

- raw corpus text
- filename
- selected chunks
- best chunking method
- LSA matrix
- fitted `TfidfVectorizer`
- fitted `TruncatedSVD`
- fitted `PCA`
- query history

The corpus is the foundation for all vector retrieval. The graph is stored separately in Neo4j Aura and is usually generated through the Colab workflow.

## 2. Chunking Module

Implemented in:

```text
app/services/chunking_service.py
```

The chunking step runs seven algorithms over the corpus and chooses the best one using a composite score.

### Chunking Algorithms

1. Fixed-size chunking

Splits the corpus into fixed character windows.

Parameters:

```python
FIXED_CHARS = 500
```

This method is simple and predictable, but it can split sentences or concepts in unnatural places.

2. Sentence chunking

First splits the text into sentences, then accumulates sentences into chunks of roughly 600 characters.

The sentence splitter uses punctuation boundaries:

```python
(?<=[.!?])\s+(?=[A-Z...])
```

This preserves sentence boundaries better than fixed-size chunking.

3. Paragraph chunking

Splits the text on blank lines:

```python
\n\s*\n
```

This keeps natural document structure when the source text has useful paragraph breaks.

4. Semantic chunking

Splits text into sentences, embeds each sentence with TF-IDF, and groups adjacent sentences if their cosine similarity to the current chunk centroid is high enough.

Core idea:

```text
current chunk centroid = mean TF-IDF vector of sentences already in chunk
similarity = cosine(centroid, next_sentence_vector)
if similarity >= 0.25: keep sentence in current chunk
else: start a new chunk
```

Parameter:

```python
SEM_THRESHOLD = 0.25
```

This is a lightweight semantic segmentation method. It does not use neural embeddings; it uses TF-IDF similarity.

5. Sliding-window chunking

Creates overlapping character windows.

Parameters:

```python
WINDOW_CHARS = 600
OVERLAP_CHARS = 100
step = 500
```

Overlap helps preserve context across chunk boundaries.

6. Recursive chunking

Recursively splits text using a hierarchy of separators:

```python
["\n\n", "\n", ". ", " "]
```

It tries to keep chunks under:

```python
RECURSIVE_MAX = 500
```

This resembles recursive text splitting in many RAG systems: split by larger structure first, then smaller separators only when needed.

7. Topic-structure chunking

Uses regex patterns that look like section headings, numbered headings, uppercase headings, or separators. If no topic structure is found, it falls back to paragraph chunking.

This is useful for thesis-like or report-like documents with numbered sections.

### Chunking Metrics

Each chunking method is evaluated by `_eval_chunks`.

Metrics:

```text
n_chunks: number of chunks
avg_len: average chunk length in characters
std_len: standard deviation of chunk length
vocab_density: unique_tokens / total_tokens
score: composite chunking score
```

The composite score is:

```text
std_norm = std(chunk_lengths) / mean(chunk_lengths)

score =
  0.5 * vocab_density
  + 0.3 * (1 - min(std_norm, 1))
  + 0.2 * min(1, n_chunks / 200)
```

Interpretation:

- `vocab_density` rewards chunks that preserve lexical variety.
- `1 - std_norm` rewards stable chunk sizes.
- `min(1, n_chunks / 200)` rewards enough chunk granularity without growing unbounded.

The best method is the one with the highest score.

Outputs:

```text
static/A1_chunking_comparison.png
static/A2_hierarchy_tree.png
```

The selected chunks become the active chunks for embedding and retrieval.

## 3. Embeddings Module

Implemented in:

```text
app/services/embeddings_service.py
```

This project does not use neural sentence embeddings. It uses a classical vector-space pipeline:

```text
chunks
  -> TF-IDF sparse vectors
  -> TruncatedSVD / LSA dense vectors
  -> L2 normalization
  -> PCA 2D for visualization
```

### TF-IDF

The backend uses `TfidfVectorizer`:

```python
TfidfVectorizer(
    max_features=12000,
    sublinear_tf=True,
    ngram_range=(1, 2),
)
```

Important details:

- `max_features=12000`: keeps the vocabulary bounded.
- `sublinear_tf=True`: replaces raw term frequency with logarithmic scaling, reducing the dominance of very frequent words.
- `ngram_range=(1, 2)`: uses unigrams and bigrams, allowing terms like `wind stress` or `surface heat` to carry meaning.

TF-IDF concept:

```text
tfidf(term, doc) = tf(term, doc) * idf(term)
```

where IDF gives more weight to terms that are informative across the corpus.

### TruncatedSVD / LSA

The TF-IDF matrix is reduced with `TruncatedSVD`.

Parameter:

```python
LSA_DIMS = 100
```

The actual number of components is:

```python
n_components = min(100, len(chunks) - 1)
```

This produces dense latent semantic vectors. In information retrieval, this is commonly called LSA: Latent Semantic Analysis.

After SVD, vectors are normalized:

```python
lsa = normalize(svd.fit_transform(tfidf))
```

Normalization matters because FAISS later uses inner product. For normalized vectors, inner product is equivalent to cosine similarity.

### PCA

PCA is used only for visualization:

```python
PCA(n_components=2)
```

It projects the LSA vectors into two dimensions and exposes explained variance ratios:

```text
PC1 variance
PC2 variance
```

These values describe how much of the LSA-space variance is visible in the 2D plot.

### KMeans Clustering

The embedding module clusters chunks in LSA space, not PCA space.

The number of clusters is:

```python
n_cl = max(2, min(8, len(chunks) // 5, len(chunks) - 1))
```

KMeans settings:

```python
KMeans(n_clusters=n_cl, random_state=42, n_init=10)
```

Purpose:

- group semantically similar chunks
- color the PCA visualization
- show global corpus structure

### Query Projection

The query is projected through the same fitted pipeline:

```text
query
  -> fitted TF-IDF vectorizer
  -> fitted SVD
  -> normalized LSA vector
  -> fitted PCA for visualization
```

Then cosine similarity is computed between the query LSA vector and all chunk LSA vectors.

The top 10 closest chunks are highlighted in:

```text
static/B1_pca2d_embeddings.png
```

## 4. Retrieval Module

Implemented in:

```text
app/services/retrieval_service.py
```

Retrieval compares five algorithms:

1. Top-k Semantic
2. Cosine TF-IDF
3. BM25
4. Hybrid BM25 + Embedding
5. MMR

Each method returns:

```text
chunk_index, score, chunk_text
```

Default:

```python
TOP_K_RETRIEVAL = 5
```

### Method 1: Top-k Semantic

This method embeds the query into normalized LSA space and searches against the normalized chunk LSA matrix.

It uses FAISS:

```python
faiss.IndexFlatIP
```

Because vectors are normalized, inner product is cosine similarity.

Formula:

```text
score(q, d) = q dot d = cosine(q, d)
```

### Method 2: Cosine TF-IDF

This method compares the raw TF-IDF query vector against raw TF-IDF chunk vectors.

Formula:

```text
score(q, d) = cosine(tfidf(q), tfidf(d))
```

This is more lexical than LSA retrieval. It is strong when the query uses terms that appear directly in the corpus.

### Method 3: BM25

BM25 is a probabilistic lexical retrieval algorithm based on term frequency, inverse document frequency, and document length normalization.

Parameters:

```python
BM25_K1 = 1.5
BM25_B = 0.75
```

For each query term:

```text
idf(term) = log((N - df + 0.5) / (df + 0.5) + 1)

score += idf(term) *
         (tf * (k1 + 1)) /
         (tf + k1 * (1 - b + b * doc_len / avg_doc_len))
```

BM25 rewards:

- term overlap with the query
- terms that are rare in the corpus
- term frequency, with saturation

It penalizes very long chunks through length normalization.

The implementation normalizes BM25 scores to `[0, 1]` before returning them.

### Method 4: Hybrid BM25 + Embedding

This method combines semantic LSA similarity and BM25 lexical similarity.

Parameter:

```python
HYBRID_ALPHA = 0.5
```

Formula:

```text
hybrid_score =
  alpha * norm01(semantic_score)
  + (1 - alpha) * norm01(BM25_score)
```

With `alpha = 0.5`, semantic and lexical scores are weighted equally.

This is the preferred method for the agentic final answer:

```python
AGENTIC_VECTOR_METHOD = "4.Hybrid BM25+Emb"
```

Reason:

- BM25 anchors retrieval in exact scientific terms.
- LSA captures softer semantic similarity.
- The combination often gives better evidence for LLM answer generation than pure lexical or pure semantic retrieval.

### Method 5: MMR

MMR means Maximal Marginal Relevance. It tries to select chunks that are both relevant and diverse.

Parameter:

```python
MMR_LAMBDA = 0.6
```

Selection rule:

```text
MMR(candidate) =
  lambda * relevance(candidate)
  - (1 - lambda) * max_similarity(candidate, already_selected)
```

With `lambda = 0.6`, relevance matters more than diversity, but repeated near-duplicates are penalized.

MMR is useful when the system needs broader coverage, though the final agentic answer currently prefers hybrid BM25 + embeddings.

### Retrieval Metrics

Every retrieval method is evaluated with:

```text
avg_score
diversity
harmonic
```

`avg_score`:

```text
mean(score of retrieved chunks)
```

This estimates average relevance according to the method's own scores.

`diversity`:

```text
diversity = 1 - mean(pairwise cosine similarity among retrieved chunk LSA vectors)
```

High diversity means the retrieved chunks are less redundant.

`harmonic`:

```text
harmonic = 2 * diversity * avg_score / (diversity + avg_score + 1e-9)
```

This is a harmonic mean of relevance and diversity. It rewards methods that balance both.

The retrieval comparison plot is saved to:

```text
static/C1_retrieval_comparison.png
```

### Extractive Summary

The retrieval service also builds a short extractive resume from the best retrieved results.

Process:

1. Split retrieved chunks into sentences.
2. Vectorize sentences plus query with TF-IDF.
3. Rank sentences by cosine similarity to query.
4. Return the top six sentences in original order.

This summary is not the final LLM answer. It is a retrieval-side artifact.

### FAISS Vector Store

The retrieval step saves:

```text
static/vector_store.faiss
static/vector_store_metadata.json
```

The FAISS index uses:

```python
IndexFlatIP
```

Metadata includes:

- number of chunks
- embedding dimension
- best chunking method
- index type
- similarity type
- chunk IDs and text

This vector store can be reused locally and packaged for Colab.

## 5. Graph RAG Layer

Implemented in:

```text
app/services/graph_service.py
```

The graph layer connects to Neo4j Aura. The graph is expected to contain:

```text
(:Entity)-[:RELATIONSHIP]->(:Entity)
```

Entity nodes may have:

- `name`
- `community`
- `mentions`

Relationships may have:

- relationship type
- `weight`

### Neo4j Configuration

Settings come from `.env`:

```text
NEO4J_URI
NEO4J_USERNAME
NEO4J_PASSWORD
NEO4J_DATABASE
```

The graph service verifies connectivity before opening a session.

### Graph Status

`graph_status()` counts:

```cypher
MATCH (e:Entity)
OPTIONAL MATCH (e)-[r]->(:Entity)
RETURN count(DISTINCT e) AS nodes, count(r) AS relationships
```

It also counts Louvain communities:

```cypher
MATCH (e:Entity)
WHERE e.community IS NOT NULL
RETURN count(DISTINCT e.community) AS community_count
```

### Graph Search

Graph search is keyword-based.

The query is converted into up to 12 keywords after removing stopwords. The search Cypher matches if a keyword appears in:

- source entity name
- target entity name
- relationship type

Core Cypher:

```cypher
MATCH (a:Entity)-[r]->(b:Entity)
WHERE any(term IN $keywords WHERE
    toLower(coalesce(a.name, '')) CONTAINS term OR
    toLower(coalesce(b.name, '')) CONTAINS term OR
    toLower(type(r)) CONTAINS term
)
RETURN a.name AS source,
       type(r) AS relationship,
       b.name AS target,
       a.community AS source_community,
       b.community AS target_community,
       coalesce(r.weight, 1) AS weight
ORDER BY weight DESC
LIMIT $limit
```

This is not graph embedding search. It is keyword filtering over symbolic graph relationships.

### Graph Answer

`graph_answer()` returns:

- graph search rows
- community summary rows
- a simple relationship string

The final LLM answer does not use this string directly as the whole answer. Instead, the agent passes graph rows into the LLM evidence formatter.

### Graph Network Endpoint

The graph network endpoint exports nodes and edges for frontend visualization.

Nodes:

```cypher
MATCH (e:Entity)
RETURN elementId(e) AS id,
       e.name AS name,
       e.community AS community,
       coalesce(e.mentions, 1) AS mentions
```

Edges:

```cypher
MATCH (a:Entity)-[r]->(b:Entity)
WHERE elementId(a) IN $node_ids AND elementId(b) IN $node_ids
RETURN elementId(a) AS source,
       elementId(b) AS target,
       type(r) AS relationship,
       coalesce(r.weight, 1) AS weight
```

The backend groups nodes by `community`, counts sizes, sums mentions, and returns examples.

## 6. Louvain Communities

Louvain community detection is performed outside the main FastAPI request path, through the Colab or Neo4j workflow.

Conceptually:

1. Build a graph where entities are nodes and extracted relationships are edges.
2. Run Louvain community detection.
3. Assign a community ID to each entity node.
4. Write the `community` property back to Neo4j.

Louvain optimizes modularity, which measures whether a graph has more within-community edges than expected by chance.

Modularity idea:

```text
Q = sum over communities of
    observed internal edge density - expected internal edge density
```

In this project, Louvain communities are used for:

- graph visualization
- community filtering
- grouping related entities
- giving the graph structure more interpretability

The FastAPI backend does not recompute Louvain on every request. It reads `community` properties already stored in Neo4j.

## 7. Query Classification

Implemented in:

```text
app/routers/query.py
```

The classifier is rule-based. It does not call an LLM.

It uses:

- conceptual terms
- relational terms
- relational phrases
- generic question terms

Examples of conceptual terms:

```text
what, explain, define, describe, summary, why, how
```

Examples of relational terms:

```text
relation, related, connect, control, influence, affect, cause, depend, contribute
```

Examples of relational phrases:

```text
depends on, based on, linked to, related to, connected to, contributes to
```

### Classification Scores

The classifier tokenizes the normalized query and computes:

```text
graph_hits = relational terms found, excluding generic question terms
vector_hits = conceptual terms found
phrase_hits = relational phrases found

graph_score = 2 * len(graph_hits) + 3 * len(phrase_hits)
vector_score = len(vector_hits)
```

Routing logic:

- If relational score is strong and conceptual terms are present: `hybrid`
- If relational score is strong alone: `systematic`
- If both graph and vector signals are present: `hybrid`
- If graph score dominates: `systematic`
- Otherwise: `semantic`

Returned routes:

```text
semantic   -> Vectorial RAG
systematic -> Graph RAG
hybrid     -> Agentic Graph-Vector RAG
```

The classifier also returns a confidence value.

## 8. Q-Learning Route Policy

Implemented in:

```text
app/services/rl_service.py
```

The Q-learning layer adapts routing based on feedback.

States:

```python
STATES = ["semantic", "systematic", "hybrid"]
```

Actions/routes:

```python
ROUTES = ["Vectorial RAG", "Graph RAG", "Agentic Graph-Vector RAG"]
```

Default policy:

```text
semantic:
  Vectorial RAG: 0.7
  Graph RAG: 0.1
  Agentic Graph-Vector RAG: 0.3

systematic:
  Vectorial RAG: 0.1
  Graph RAG: 0.7
  Agentic Graph-Vector RAG: 0.3

hybrid:
  Vectorial RAG: 0.3
  Graph RAG: 0.3
  Agentic Graph-Vector RAG: 0.7
```

### Route Choice

The classifier provides a fallback route. Q-learning can override it only if the learned best route is better by a margin:

```python
POLICY_MARGIN = 0.05
```

Logic:

```text
best_route = argmax Q(state, route)
if Q(state, best_route) >= Q(state, fallback_route) + margin:
    choose best_route
else:
    choose fallback_route
```

This avoids unstable route switching when learned scores are close.

### Feedback Update

Hyperparameters:

```python
ALPHA = 0.35
GAMMA = 0.75
```

Reward is clipped to:

```text
[-1, 1]
```

Update formula:

```text
td_target = reward + gamma * max_a Q(next_state, a)
td_error = td_target - Q(state, route)
Q(state, route) = Q(state, route) + alpha * td_error
```

The Q-table is persisted in:

```text
static/rl_policy.json
```

Feedback endpoint:

```text
POST /agentic/feedback
```

Payload:

```json
{
  "query": "What controls tropical Pacific warming?",
  "route": "Agentic Graph-Vector RAG",
  "reward": 1
}
```

## 9. Agentic Orchestration

Implemented in:

```text
app/services/agentic_service.py
```

Main endpoint:

```text
POST /agentic/ask
```

The high-level agent flow is:

```text
validate query
  -> classify query
  -> choose route through Q-learning
  -> check response cache
  -> retrieve vector evidence if route needs it
  -> retrieve graph evidence if route needs it
  -> synthesize final answer with LLM
  -> cache successful answer
  -> return answer and evidence metadata
```

### Query Validation

Before retrieval, the agent validates the query.

Validation types:

1. unclear or ungrammatical
2. semantically incoherent
3. out of domain

#### Unclear or Ungrammatical

The agent rejects very short queries, malformed patterns, trailing connectors, and questions without enough verb structure.

Examples of malformed patterns:

```text
and no
small and no
no during
do the when
```

Response:

```text
The query contains corpus terms, but it is not grammatically or semantically clear enough to answer. Please reformulate it as a complete question.
```

#### Semantically Incoherent

The agent checks whether climate concepts are combined with impossible properties.

Climate concepts include:

```text
Walker circulation, SST, gradient, CMIP, models, Pacific, warming, ENSO, wind, heat, ocean
```

Impossible properties include:

```text
angry, happy, dream, taste, smell, speak, listen, emotion, intention
```

If both appear, the query is marked `semantically_incoherent`.

#### Out of Domain

The agent checks whether the query contains project domain terms such as:

```text
Pacific, tropical, climate, warming, SST, ocean, atmosphere, ENSO, Walker, CMIP, model, bias, graph, FAISS, RAG
```

If not, it returns an out-of-corpus response.

### Mechanism Query Expansion

For broad mechanism questions about tropical Pacific warming, the agent expands the retrieval query with mechanism terms:

```text
surface heat fluxes
wind stress
freshwater forcing
east-west SST gradient
trade winds
mean-state biases
equatorial warming
Southeast Pacific cooling
```

This is done only for vector retrieval. The original user query remains unchanged for the final answer.

Purpose:

- improve retrieval recall
- pull in mechanism-focused chunks
- reduce generic answers

### Route Execution

If route is:

```text
Vectorial RAG
```

the agent retrieves vector evidence only.

If route is:

```text
Graph RAG
```

the agent currently still retrieves vector evidence, then retrieves graph evidence. This gives the LLM more grounding while preserving graph-oriented routing.

If route is:

```text
Agentic Graph-Vector RAG
```

the agent retrieves both vector and graph evidence.

### Agentic Cache

Successful LLM responses are cached in:

```text
static/agentic_response_cache.json
```

The cache key includes:

- normalized query
- cache version
- LLM provider
- active model name
- vector metadata hash
- agentic vector retrieval method

The metadata hash invalidates cached answers when the vector store changes.

Only successful LLM answers are cached. Validation failures and model failure messages are not cached as successful LLM synthesis.

## 10. LLM Synthesis

Implemented in:

```text
app/services/llm_service.py
```

The LLM service takes:

- user query
- selected route
- vector retrieval data
- graph retrieval data

and returns:

```text
final answer, used_llm_boolean
```

### Evidence Formatting

The LLM does not receive raw service objects. The backend formats evidence into concise text sections.

Vector evidence:

- takes top retrieval results
- splits them into sentences
- filters by likely query language
- selects up to four useful sentences

Graph evidence:

- formats graph relationships as triples
- filters generic noisy nodes
- skips noisy `CO_OCCURS_WITH` relationships

Related entities:

- finds graph entities related to query terms
- deduplicates entities
- returns up to 12 names

Control factors:

The service scans evidence for known mechanism terms and creates a compact list of control factors, such as:

- sea-surface temperature patterns
- Walker circulation
- Bjerknes feedback
- east-west ocean-atmosphere gradient
- equatorial ocean dynamics
- surface heat fluxes
- mean-state biases in climate models

### Prompt Structure

The final prompt includes:

- semantic coherence instruction
- user question
- graph relational evidence
- candidate graph entities
- detected control factors
- vector evidence passages
- answer instructions

The prompt tells the model:

- answer in the query language
- do not repeat the question
- do not mention internal stores or routing
- use only the provided evidence
- explain mechanisms when needed
- say when evidence is insufficient
- write one natural paragraph

### Model Providers

Supported providers:

```text
gemini
local / transformers / huggingface
```

Current recommended provider:

```text
gemini
```

Gemini settings:

```python
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_MAX_OUTPUT_TOKENS = 512
GEMINI_RETRY_MAX_OUTPUT_TOKENS = 384
GEMINI_THINKING_BUDGET = 0
```

If Gemini fails, the backend logs the exception and returns:

```text
gemini-2.5-flash could not generate a reliable answer for this query.
```

A Gemini `503 UNAVAILABLE` means the model endpoint is temporarily under high demand. A quota problem usually appears as `429`, `ResourceExhausted`, or quota/rate-limit wording.

### Output Validation

The backend validates model output for:

- empty answer
- answer too short
- query echo
- severe repetition
- prompt-label leakage
- mixed-language output for French queries
- weak control/mechanism answer
- incomplete-looking endings

The validation was intentionally designed to avoid returning broken local-model generations. For Gemini, it should accept concise mechanism answers if they contain enough scientific mechanism terms.

If the first answer is invalid, the service builds a shorter retry prompt and calls the model again.

If retry also fails validation, the backend returns the reliable-failure message rather than inventing a deterministic fallback answer.

## 11. Debug Endpoint

Endpoint:

```text
POST /agentic/debug
```

This runs the agent up to evidence assembly without calling the LLM.

It returns:

- detected type
- classifier route
- learned route
- policy scores
- vector method
- vector results
- graph results
- communities
- formatted vector evidence
- formatted graph evidence
- related entities
- control factors
- final prompt
- prompt length

This endpoint is important for diagnosing whether a bad answer is caused by:

- bad routing
- poor vector evidence
- noisy graph evidence
- prompt construction
- LLM generation
- output validation

## 12. Full Backend API Flow

A normal local pipeline run is:

```text
POST /chunking
POST /embeddings
POST /retrieval
POST /agentic/ask
```

### `/chunking`

Runs all chunking methods, stores the best chunks, and exports chunking plots.

### `/embeddings`

Fits TF-IDF, SVD, PCA on current chunks, stores fitted objects, computes query projection, clusters chunks, and exports the embedding plot.

### `/retrieval`

Runs all retrieval methods, evaluates metrics, exports comparison plot, saves FAISS index and metadata.

### `/agentic/ask`

Validates query, classifies query, chooses route, retrieves evidence, calls LLM, validates output, caches successful response.

## 13. Important Metrics Summary

### Chunking Metrics

```text
n_chunks
avg_len
std_len
vocab_density = unique_tokens / total_tokens
std_norm = std_len / avg_len
score = 0.5*density + 0.3*(1-min(std_norm,1)) + 0.2*min(1,n_chunks/200)
```

### Embedding Metrics

```text
explained_variance[0] = PCA PC1 variance ratio
explained_variance[1] = PCA PC2 variance ratio
cluster_labels = KMeans labels on LSA vectors
top_k_indices = chunks closest to query by cosine similarity
```

### Retrieval Metrics

```text
avg_score = mean retrieval score
diversity = 1 - mean pairwise cosine similarity among selected chunks
harmonic = 2 * avg_score * diversity / (avg_score + diversity + 1e-9)
```

### Q-Learning Metrics

```text
td_target = reward + gamma * max(Q(next_state,*))
td_error = td_target - Q(state, route)
updated_q = old_q + alpha * td_error
```

Returned feedback metadata includes:

- state
- next state
- route
- reward
- alpha
- gamma
- TD target
- TD error
- updated Q-value
- policy scores

## 14. Why This Is Agentic

The system is agentic because it does more than retrieve chunks and answer.

It:

1. Validates whether the query is answerable.
2. Classifies the query type.
3. Uses a learned policy to choose a route.
4. Chooses vector evidence, graph evidence, or both.
5. Expands mechanism queries for better recall.
6. Formats evidence differently depending on language and route.
7. Calls an LLM with structured evidence.
8. Validates the generated answer.
9. Stores successful answers in a persistent cache.
10. Updates route preferences from user feedback.

The agent is still mostly deterministic and rule-based. The LLM is used for final natural-language synthesis, not for deciding every step.

## 15. Strengths and Limitations

### Strengths

- Transparent algorithms: TF-IDF, SVD, BM25, MMR, KMeans, Q-learning.
- Local vector store with FAISS.
- Graph relationships available through Neo4j.
- Hybrid route can combine lexical, semantic, and symbolic evidence.
- Debug endpoint exposes prompt and evidence.
- Feedback updates route policy.
- Persistent response cache reduces repeated LLM calls.

### Limitations

- Embeddings are classical TF-IDF/LSA, not neural sentence embeddings.
- Graph search is keyword-based, not graph-embedding search.
- Graph quality depends on extraction quality from the Colab workflow.
- Query classification is rule-based and can misclassify edge cases.
- Q-learning has only three states and three actions, so policy adaptation is simple.
- LLM answer quality depends on Gemini availability and quota/service status.
- Output validation can reject answers if rules are too strict, so validator tuning matters.

## 16. Mental Model

The backend can be understood as three cooperating retrieval systems:

```text
Vector RAG:
  Finds relevant corpus passages using TF-IDF, LSA, BM25, hybrid search, and FAISS.

Graph RAG:
  Finds symbolic entity relationships and communities in Neo4j.

Agentic RAG:
  Decides which retrieval route to use, fuses evidence, calls the LLM, validates output, and learns from feedback.
```

The final answer is generated only after the backend has assembled grounded evidence. The main goal is not just to answer, but to answer from the corpus and graph with a route selected by the agent.
