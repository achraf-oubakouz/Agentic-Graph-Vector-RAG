# Agentic Graph-Vector RAG System Documentation

## 1. Project Overview

This project is an **Agentic Graph-Vector Retrieval-Augmented Generation system** built around a scientific corpus about **tropical Pacific warming, CMIP climate models, SST gradients, model biases, ocean-atmosphere dynamics, and related climate mechanisms**.

The system combines:

- **Vector RAG** using chunking, embeddings, dimensionality reduction, and **FAISS**.
- **Graph RAG** using extracted entities/relationships stored in **Neo4j Aura**.
- **Louvain community detection** for graph clustering.
- **Agentic routing** that chooses between Vector RAG, Graph RAG, or Hybrid Graph-Vector RAG.
- **Q-learning** for route-policy adaptation based on user feedback.
- **LLM final response generation** from retrieved vector/graph evidence.
- **React frontend dashboard** for running the pipeline, asking questions, visualizing metrics, and viewing the graph/communities.

The project follows the requested pipeline:

```text
User question
  -> Agent orchestrator
  -> Query classification
  -> Q-learning route policy
  -> Vector Store and/or Graph Store
  -> Evidence fusion
  -> LLM final answer
  -> User feedback
  -> Q-table update
```

## 2. Main Technologies

### Backend

- **Python**
- **FastAPI**: backend API framework.
- **Uvicorn**: ASGI server for FastAPI.
- **Pydantic / pydantic-settings**: request/response models and configuration.
- **Neo4j Python Driver**: connects to Neo4j Aura.
- **FAISS CPU**: local vector index.
- **scikit-learn**: TF-IDF, TruncatedSVD, PCA, cosine similarity.
- **NumPy / pandas**: numerical processing and metric tables.
- **Matplotlib**: exports metric and visualization images.
- **Google GenAI SDK**: Gemini API final-response generation.
- **Transformers + PyTorch + sentencepiece**: optional local Hugging Face provider.

### Frontend

- **React**
- **Vite**
- **lucide-react** icons
- **react-force-graph-2d** for interactive graph visualization
- Standard CSS in `frontend/src/styles.css`

### Graph / Cloud / Notebook

- **Neo4j Aura**: cloud graph database.
- **Google Colab**: used for graph generation and Louvain processing.
- **NetworkX / Louvain workflow in Colab scripts**: graph community detection.

### Containerization / Deployment

- **Docker**: backend and frontend container images.
- **Nginx**: serves the frontend production build and proxies `/api/*` requests to the backend.
- **Kubernetes**: backend and frontend Deployments plus NodePort Services under `k8s/`.

## 3. Current LLM Configuration

The current final-response model is configured in:

```text
app/core/config.py
```

Current active provider:

```python
LLM_PROVIDER = "gemini"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
```

Final response generation is currently configured to use the **Gemini API** through the official `google-genai` SDK. Retrieval, routing, FAISS, Neo4j, and evidence formatting remain local.

Important:

- The recommended active model is `gemini-2.5-flash`.
- `GEMINI_API_KEY` must be set in `.env`.
- Gemini thinking is disabled for this short RAG synthesis task with `GEMINI_THINKING_BUDGET = 0`, so output tokens are used for the visible answer.
- Local Hugging Face generation is still present behind `LLM_PROVIDER = "local"`, but `mistralai/Mistral-7B-Instruct-v0.3` is not practical on this Intel Iris Xe laptop because it requires CPU inference and produces very slow/truncated outputs.

## 4. Project Structure

```text
C:\Projects\PFA
├── main.py
├── requirements.txt
├── PROJECT_DOCUMENTATION.md
├── README.md
├── Fiche Explicative2-Seance5-.pdf
├── data
│   └── data_final_cleaned.txt
├── static
│   ├── vector_store.faiss
│   ├── vector_store_metadata.json
│   ├── rl_policy.json
│   ├── agentic_response_cache.json
│   ├── colab_graph_rag_package.zip
│   ├── A1_chunking_comparison.png
│   ├── A2_hierarchy_tree.png
│   ├── B1_pca2d_embeddings.png
│   └── C1_retrieval_comparison.png
├── app
│   ├── core
│   │   ├── config.py
│   │   ├── database.py
│   │   └── security.py
│   ├── routers
│   │   ├── corpus.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   ├── retrieval.py
│   │   ├── query.py
│   │   ├── graph.py
│   │   ├── agentic.py
│   │   └── colab.py
│   ├── services
│   │   ├── chunking_service.py
│   │   ├── embeddings_service.py
│   │   ├── retrieval_service.py
│   │   ├── graph_service.py
│   │   ├── agentic_service.py
│   │   ├── llm_service.py
│   │   └── rl_service.py
│   └── schemas
│       └── models.py
├── colab
│   ├── README.md
│   ├── agentic_graph_vector_rag_colab.ipynb
│   ├── graph_rag_colab_pipeline.py
│   ├── louvain_existing_neo4j.ipynb
│   └── louvain_existing_neo4j.py
└── frontend
    ├── package.json
    ├── index.html
    └── src
        ├── main.jsx
        └── styles.css
```

## 5. How To Start The App

This section describes the local development workflow. Containerized and Kubernetes deployment are documented later in section 30.

### Start Backend

```powershell
cd C:\Projects\PFA
venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Backend URL:

```text
http://127.0.0.1:8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

### Start Frontend

```powershell
cd C:\Projects\PFA\frontend
npm.cmd run dev
```

Frontend URL:

```text
http://127.0.0.1:5173
```

## 6. Configuration

Configuration is defined in:

```text
app/core/config.py
```

Important settings:

```python
STATIC_DIR = "static"
FAISS_INDEX_PATH = "static/vector_store.faiss"
FAISS_METADATA_PATH = "static/vector_store_metadata.json"
CORPUS_PATH = "data/data_final_cleaned.txt"
COLAB_PACKAGE_PATH = "static/colab_graph_rag_package.zip"
RL_POLICY_PATH = "static/rl_policy.json"
AGENTIC_CACHE_PATH = "static/agentic_response_cache.json"
ENABLE_LLM_SYNTHESIS = True
LLM_PROVIDER = "gemini"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
GEMINI_API_KEY = ""
GEMINI_THINKING_BUDGET = 0
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
```

Neo4j settings are loaded from `.env`:

```text
NEO4J_URI=neo4j+s://xxxx.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=neo4j
```

Gemini settings are also loaded from `.env`:

```text
LLM_PROVIDER=gemini
GEMINI_MODEL_NAME=gemini-2.5-flash
GEMINI_API_KEY=...
ENABLE_LLM_SYNTHESIS=True
```

Do not commit real credentials.

## 7. Corpus Loading

The corpus is automatically loaded on FastAPI startup from:

```text
data/data_final_cleaned.txt
```

The file must have exactly this path/name unless `CORPUS_PATH` is changed.

The in-memory corpus store is defined in:

```text
app/core/database.py
```

The shared store contains:

- `text`
- `filename`
- `chunks`
- `best_chunking_method`
- embedding objects
- query history

## 8. API Routers

Routers are registered in `main.py`.

### Root

```text
GET /
```

Returns project name, version, docs URL, corpus status, and vector store type.

### Corpus

```text
GET /corpus/status
POST /corpus/upload
```

Checks whether corpus text is loaded or uploads a corpus file.

### Chunking

```text
POST /chunking
```

Runs all chunking methods and selects the best chunking strategy.

### Embeddings

```text
POST /embeddings
```

Requires JSON body:

```json
{
  "query": "What controls tropical Pacific warming?"
}
```

Builds vector embeddings and PCA visualization.

### Retrieval

```text
POST /retrieval
```

Requires JSON body:

```json
{
  "query": "What controls tropical Pacific warming?"
}
```

Runs retrieval methods and saves FAISS vector store.

### Query Router

```text
POST /query-router
GET /query-router/history
```

Classifies the query as:

- `semantic`
- `systematic`
- `hybrid`

And suggests:

- `Vectorial RAG`
- `Graph RAG`
- `Agentic Graph-Vector RAG`

### Graph

```text
GET /graph/status
POST /graph/search
GET /graph/network
```

Connects to Neo4j Aura, searches graph relationships, and returns graph network data for the frontend.

### Agentic

```text
POST /agentic/ask
POST /agentic/debug
POST /agentic/feedback
GET /agentic/policy
```

This is the main Agentic Graph-Vector RAG interface.

`POST /agentic/debug` returns the detected route, vector results, graph results, formatted LLM evidence, control factors, related entities, and the final prompt without calling the LLM. Use it to audit retrieval quality before testing final generation.

### Colab

```text
GET /colab/status
POST /colab/package
```

Builds a ZIP package for Google Colab containing:

- FAISS index
- vector metadata
- Colab notebook
- Colab scripts
- README

## 9. Full Local Pipeline

The pipeline can be run from the frontend or through API calls.

Manual API flow:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chunking
```

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/embeddings `
  -ContentType "application/json" `
  -Body '{"query":"What controls tropical Pacific warming?"}'
```

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/retrieval `
  -ContentType "application/json" `
  -Body '{"query":"What controls tropical Pacific warming?"}'
```

The retrieval step regenerates:

```text
static/vector_store.faiss
static/vector_store_metadata.json
```

## 10. Chunking Module

Implemented in:

```text
app/services/chunking_service.py
```

The project evaluates several chunking methods:

1. Fixed-size
2. Sentences
3. Paragraphs
4. Semantic
5. Sliding-Window
6. Recursive
7. Topic-Structure

For each method, metrics include:

- number of chunks
- average chunk length
- standard deviation of length
- vocabulary density
- score

Outputs:

```text
static/A1_chunking_comparison.png
static/A2_hierarchy_tree.png
```

## 11. Embeddings Module

Implemented in:

```text
app/services/embeddings_service.py
```

The embedding pipeline uses:

- `TfidfVectorizer`
- `TruncatedSVD`
- normalization
- PCA visualization
- clustering labels
- top query neighbors

This is not a transformer embedding model. It is a classical vector-space embedding pipeline based on TF-IDF + dimensionality reduction.

Output:

```text
static/B1_pca2d_embeddings.png
```

## 12. FAISS Vector Store

Implemented in:

```text
app/services/retrieval_service.py
```

FAISS index:

```text
static/vector_store.faiss
```

Metadata:

```text
static/vector_store_metadata.json
```

The FAISS index uses:

```text
IndexFlatIP
```

Similarity:

```text
cosine similarity over normalized vectors
```

The FAISS vector store is also packaged for Colab.

## 13. Retrieval Methods

The retrieval module compares five methods:

1. `Top-k Semantic`
2. `Cosine TF-IDF`
3. `BM25`
4. `Hybrid BM25+Emb`
5. `MMR`

Metrics:

- `avg_score`: relevance score
- `diversity`: result diversity
- `harmonic`: harmonic combination of relevance and diversity

Important distinction:

- `/retrieval` still evaluates and reports the metric-selected best method.
- `/agentic/ask` currently uses:

```text
4.Hybrid BM25+Emb
```

for final LLM evidence because it tends to be better for answer generation than pure BM25 or very diverse MMR.

The current agentic vector method is defined in:

```text
app/services/agentic_service.py
```

```python
AGENTIC_VECTOR_METHOD = "4.Hybrid BM25+Emb"
```

## 14. Graph Store With Neo4j Aura

Graph code is in:

```text
app/services/graph_service.py
```

The graph stores extracted entities as Neo4j nodes and extracted relations as Neo4j relationships.

The app expects Neo4j nodes to include properties such as:

- `name`
- `community`
- `mentions`

The frontend graph uses:

- nodes
- edges
- relationship labels
- Louvain community IDs
- community filters

Graph search endpoint:

```text
POST /graph/search
```

Example:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/graph/search `
  -ContentType "application/json" `
  -Body '{"query":"Walker circulation","limit":20}'
```

## 15. Google Colab Graph Generation

The Colab workflow is under:

```text
colab/
```

Main files:

```text
colab/agentic_graph_vector_rag_colab.ipynb
colab/graph_rag_colab_pipeline.py
colab/louvain_existing_neo4j.ipynb
colab/louvain_existing_neo4j.py
```

Local packaging endpoint:

```text
POST /colab/package
```

PowerShell:

```powershell
cd C:\Projects\PFA
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/colab/package `
  -OutFile .\static\colab_graph_rag_package.zip
```

The ZIP contains:

- `vector_store.faiss`
- `vector_store_metadata.json`
- Colab notebook
- Colab graph pipeline script
- Colab README

In Colab, the pipeline:

1. Loads FAISS metadata.
2. Uses an LLM or fallback extraction to extract entities and relationships.
3. Generates graph store JSON.
4. Generates Neo4j Cypher import.
5. Imports into Neo4j Aura.
6. Runs or supports Louvain community detection.

## 16. Louvain Communities

Louvain is used to detect clusters/communities in the graph.

The community workflow can:

- read an existing Neo4j Aura graph
- build a NetworkX graph
- run Louvain
- write `community` properties back to Neo4j nodes
- export HTML network visualizations

The frontend uses these `community` values to:

- filter communities
- color nodes
- show community counts
- display static community circles

## 17. Agentic Orchestration

The main orchestration is in:

```text
app/services/agentic_service.py
```

The flow:

1. User sends a query to:

```text
POST /agentic/ask
```

2. Agent validation runs:

- semantic incoherence validation
- grammar/clarity validation
- domain validation

3. Query is classified:

```text
semantic
systematic
hybrid
```

4. Q-learning policy chooses route:

```text
Vectorial RAG
Graph RAG
Agentic Graph-Vector RAG
```

5. Evidence is retrieved:

- vector evidence through Hybrid BM25+Emb
- graph evidence through Neo4j when needed
- broad tropical-Pacific control queries are expanded with mechanism terms such as heat fluxes, wind stress, trade winds, east-west SST gradient, mean-state biases, equatorial warming, and Southeast Pacific cooling to retrieve more mechanism-focused passages

6. LLM receives fused evidence and generates a final answer.

7. Response is cached if LLM synthesis succeeds.

## 18. Query Classification

Implemented in:

```text
app/routers/query.py
```

The classifier uses rule-based lexical logic.

Conceptual/semantic terms:

- `what`
- `explain`
- `define`
- `describe`
- `summary`
- `why`
- `how`

Relational/systematic terms:

- `relation`
- `related`
- `connected`
- `control`
- `influence`
- `affect`
- `cause`
- `depend`
- `contribute`

Possible outputs:

```text
semantic -> Vectorial RAG
systematic -> Graph RAG
hybrid -> Agentic Graph-Vector RAG
```

## 19. Q-Learning

Implemented in:

```text
app/services/rl_service.py
```

Current routes/actions:

```python
ROUTES = ["Vectorial RAG", "Graph RAG", "Agentic Graph-Vector RAG"]
```

Current states:

```python
STATES = ["semantic", "systematic", "hybrid"]
```

The Q-table is stored in:

```text
static/rl_policy.json
```

Current update formula:

```python
Q(s,a) = Q(s,a) + alpha * (reward + gamma * max(Q(s_next,*)) - Q(s,a))
```

Current hyperparameters:

```python
ALPHA = 0.35
GAMMA = 0.75
POLICY_MARGIN = 0.05
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

Rewards:

- `1`: good answer/route
- `-1`: bad answer/route

Returned values include:

- `state`
- `next_state`
- `route`
- `reward`
- `alpha`
- `gamma`
- `td_target`
- `td_error`
- `updated_q_value`
- `policy_scores`

## 20. Agent Validation

Before retrieval and LLM generation, the agent validates the query.

Implemented in:

```text
app/services/agentic_service.py
```

### Semantic Incoherence Validation

Catches questions that use valid corpus terms but ask impossible things.

Example:

```text
Why does ENSO get angry when the SST gradient starts singing?
```

Response:

```text
The query uses terms from the corpus, but it is semantically incoherent. Climate concepts do not have sensory, emotional, or intentional properties.
```

### Grammar / Clarity Validation

Catches malformed or incomplete questions.

Example:

```text
Why is SST gradient small and no during CMIP projections?
```

Response:

```text
The query contains corpus terms, but it is not grammatically or semantically clear enough to answer. Please reformulate it as a complete question.
```

### Domain Validation

Catches out-of-corpus questions.

Example:

```text
Who is the president of the USA?
```

Response says the RAG system is limited to the project corpus.

## 21. LLM Prompting

Prompt construction is in:

```text
app/services/llm_service.py
```

The prompt includes:

- semantic coherence instruction
- user question
- graph relationship evidence
- candidate graph entities
- detected control factors
- vector retrieval passages
- answer instructions

The prompt asks the LLM to:

- answer only in the query language
- not repeat the question
- not mention routing/model/Neo4j/FAISS/internal stores
- name relevant entities when needed
- explain mechanisms when needed
- write one natural paragraph
- avoid continuing abstracts or citing article metadata

The LLM output is validated for:

- empty output
- question echo
- prompt-label leakage
- bad repetition
- mixed language for French queries
- incomplete endings
- very weak answers

The project currently does **not** use a deterministic fallback answer for final responses. If the LLM fails validation or the API call fails, the response is:

```text
<model-name> could not generate a reliable answer for this query.
```

Important:

- The system should return an LLM-generated response or a clear model-failure message, not a handcrafted fallback answer.
- Gemini generation logs include finish reason, candidate token count, thought token count, character count, and a short preview.
- Generic graph nodes and noisy `CO_OCCURS_WITH` graph evidence are filtered from the LLM prompt to avoid polluting final synthesis.

## 22. Persistent Response Cache

The agentic response cache is stored in:

```text
static/agentic_response_cache.json
```

Purpose:

- Avoid repeated LLM calls for the same query.
- Keep answers after FastAPI restarts.
- Reduce latency after a query has already been answered.

Cache key includes:

- normalized query
- LLM provider
- LLM model name
- vector metadata content hash
- agentic vector retrieval method
- cache version

Cache is invalidated when:

- LLM provider changes
- LLM model changes
- vector metadata changes
- cache version changes
- agentic vector retrieval strategy changes

Only successful LLM answers are cached.

`/agentic/ask` responses include:

```json
"cached": true
```

or:

```json
"cached": false
```

## 23. Frontend Features

Frontend files:

```text
frontend/src/main.jsx
frontend/src/styles.css
```

The frontend includes:

- chat-first home page
- collapsible sidebar navigation
- fixed query composer
- inline previous-query suggestions above the query composer
- agent response panel with feedback controls
- contextual right rail with corpus snapshot and current answer metadata
- pipeline view for vector store, Colab package, and graph store controls
- improved metrics view with separated chunking, embedding, and retrieval panels
- evidence view for vector and graph evidence
- Neo4j graph visualization with Louvain community controls
- Q-learning policy view

The frontend uses:

```javascript
const API_BASE = '/api';
```

This means browser requests are sent to `/api/...`. In local development, the frontend expects a proxy or compatible serving setup. In the containerized frontend, Nginx proxies `/api/*` to the Kubernetes backend service.

The graph visualization uses:

```text
react-force-graph-2d
```

The graph has been tuned to avoid lag:

- no force animation
- no particles
- fixed node positions
- community circles
- limited labels in full graph
- all relationship labels visible in selected community view

## 24. Frontend Graph Behavior

The frontend graph supports:

- full graph view
- community-filtered view
- node labels
- relationship labels
- static community circles
- zoom/pan
- fit graph button

Full graph:

- shows all communities as one global circular visualization
- separates communities slightly
- reduces label noise

Single community:

- displays selected community as one readable static circle
- shows node names
- shows all relationship labels

## 25. Important Generated Files

### Vector Store

```text
static/vector_store.faiss
static/vector_store_metadata.json
```

Generated by `/retrieval`.

### Metrics Images

```text
static/A1_chunking_comparison.png
static/A2_hierarchy_tree.png
static/B1_pca2d_embeddings.png
static/C1_retrieval_comparison.png
```

Displayed in the frontend.

### RL Policy

```text
static/rl_policy.json
```

Stores Q-values.

### Agentic Cache

```text
static/agentic_response_cache.json
```

Stores successful LLM-generated answers.

### Colab Package

```text
static/colab_graph_rag_package.zip
```

Used to regenerate graph data in Colab.

## 26. Common Test Queries

### Vector RAG

```text
Explain tropical Pacific warming.
```

Expected route:

```text
Vectorial RAG
```

### Hybrid Agentic RAG

```text
What controls tropical Pacific warming and explain why?
```

Expected route:

```text
Agentic Graph-Vector RAG
```

### Graph RAG

```text
Which entities are related to the Walker circulation?
```

Expected route:

```text
Graph RAG
```

or:

```text
Agentic Graph-Vector RAG
```

depending on Q-learning policy.

### Mean-State Biases

```text
How do mean-state biases influence the east-west SST gradient in CMIP projections?
```

Expected answer should mention:

- mean-state biases
- east-west SST gradient
- equatorial warming
- model projection uncertainty

### Semantically Incoherent Query

```text
Why does ENSO get angry when the SST gradient starts singing?
```

Expected route:

```text
Agent validation
```

Expected response:

```text
The query uses terms from the corpus, but it is semantically incoherent. Climate concepts do not have sensory, emotional, or intentional properties.
```

### Grammatically Unclear Query

```text
Why is SST gradient small and no during CMIP projections?
```

Expected route:

```text
Agent validation
```

Expected response:

```text
The query contains corpus terms, but it is not grammatically or semantically clear enough to answer. Please reformulate it as a complete question.
```

### Out-of-Corpus Query

```text
Who is the president of the USA?
```

Expected route:

```text
Out of corpus
```

## 27. Known Limitations

### LLM Quality

Final response generation currently uses `gemini-2.5-flash` through the Gemini API. This is much more reliable on the current laptop than local `mistralai/Mistral-7B-Instruct-v0.3`, because the laptop has Intel Iris Xe graphics rather than an NVIDIA CUDA GPU.

Known considerations:

- Gemini API requires `GEMINI_API_KEY`.
- Free-tier/API quota limits can apply.
- If Gemini fails validation or the API call fails, the app returns a clear model-failure message instead of a deterministic fallback answer.
- Local Mistral 7B via Transformers is not recommended on this machine because CPU generation can be extremely slow and may truncate outputs.

### Embeddings

The current embedding pipeline uses TF-IDF + SVD, not a neural sentence embedding model.

This is acceptable for the project demo but weaker than:

- `sentence-transformers`
- `bge`
- `e5`
- MiniLM embeddings

### Graph Quality

Graph quality depends on the Colab extraction step.

Known graph noise can include generic entities such as:

- `FIGURE`
- `TABLE`
- `Cette`
- `Les`
- citation fragments

The frontend and LLM formatting try to ignore the worst generic nodes, but graph quality still depends heavily on extraction quality.

### Retrieval Metrics

BM25 may score highest by automatic metrics because it has strong keyword overlap.

However, answer generation often benefits more from:

```text
Hybrid BM25+Emb
```

because it balances lexical relevance and semantic evidence.

## 28. Neo4j Aura Troubleshooting

If you see:

```text
Failed to DNS resolve address xxxx.databases.neo4j.io:7687
```

Check:

1. Internet connection.
2. Aura instance is running.
3. `.env` URI uses:

```text
neo4j+s://xxxx.databases.neo4j.io
```

4. DNS can resolve:

```powershell
nslookup xxxx.databases.neo4j.io
```

5. Restart backend after `.env` changes.

## 29. Regenerating FAISS Vector Store

If the corpus changes, rerun:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/chunking
```

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/embeddings `
  -ContentType "application/json" `
  -Body '{"query":"What controls tropical Pacific warming?"}'
```

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/retrieval `
  -ContentType "application/json" `
  -Body '{"query":"What controls tropical Pacific warming?"}'
```

Then rebuild Colab ZIP if the graph must be regenerated:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/colab/package `
  -OutFile .\static\colab_graph_rag_package.zip
```

## 30. Containerization and Kubernetes

The project now includes Docker and Kubernetes deployment files in addition to the local development workflow.

### Backend Dockerfile

Backend container file:

```text
Dockerfile
```

The backend image:

- uses `python:3.10`
- sets `/app` as the working directory
- installs `requirements.txt`
- copies the backend project files
- starts FastAPI with Uvicorn on `0.0.0.0:8000`

Build command:

```powershell
docker build -t rag-backend:latest .
```

The container command is:

```text
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Frontend Dockerfile and Nginx

Frontend container files:

```text
frontend/Dockerfile
frontend/nginx.conf
```

The frontend image uses a multi-stage build:

1. `node:18` installs dependencies and runs `npm run build`.
2. `nginx:alpine` serves the generated `dist/` folder.

Build command:

```powershell
cd C:\Projects\PFA\frontend
docker build -t rag-frontend:latest .
```

Nginx serves the React app from:

```text
/usr/share/nginx/html
```

The Nginx config also proxies frontend API calls:

```text
/api/* -> http://backend-service:8000/*
```

This matches the frontend code, which uses `API_BASE = '/api'`.

### Kubernetes Manifests

Kubernetes files:

```text
k8s/backend-deployment.yaml
k8s/frontend-deployment.yaml
```

Backend Kubernetes resources:

- Deployment name: `rag-backend`
- Container name: `backend`
- Image: `rag-backend:latest`
- Image pull policy: `Never`
- Container port: `8000`
- Service name: `backend-service`
- Service type: `NodePort`
- NodePort: `30007`

Frontend Kubernetes resources:

- Deployment name: `rag-frontend`
- Container name: `frontend`
- Image: `rag-frontend:latest`
- Image pull policy: `Never`
- Container port: `80`
- Service name: `frontend-service`
- Service type: `NodePort`
- NodePort: `30008`

The `imagePullPolicy: Never` setting means Kubernetes expects the images to already exist locally, which is suitable for local clusters such as Docker Desktop Kubernetes or Minikube. For a remote cluster, push the images to a registry and update the image names and pull policy.

### Deploying Locally

Build both images:

```powershell
cd C:\Projects\PFA
docker build -t rag-backend:latest .

cd C:\Projects\PFA\frontend
docker build -t rag-frontend:latest .
```

Apply the Kubernetes manifests:

```powershell
cd C:\Projects\PFA
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml
```

Check resources:

```powershell
kubectl get pods
kubectl get services
```

Local NodePort URLs:

```text
Backend:  http://localhost:30007
Frontend: http://localhost:30008
```

Depending on the local Kubernetes provider, the NodePort host may be `localhost`, the Docker Desktop Kubernetes node, or the Minikube IP. For Minikube, use:

```powershell
minikube service frontend-service
```

### Environment Variables in Kubernetes

The backend manifest currently passes runtime settings through environment variables:

- `NEO4J_URI`
- `NEO4J_USERNAME`
- `NEO4J_PASSWORD`
- `NEO4J_DATABASE`
- `GEMINI_API_KEY`
- `LLM_PROVIDER`
- `GEMINI_MODEL_NAME`
- `ENABLE_LLM_SYNTHESIS`

Important security note:

- Do not commit real Neo4j credentials or Gemini API keys in Kubernetes YAML.
- Prefer Kubernetes `Secret` objects for sensitive values.
- Use `ConfigMap` objects for non-sensitive settings such as provider names, model names, and feature flags.

Recommended production direction:

```text
Secret:    NEO4J_PASSWORD, GEMINI_API_KEY
ConfigMap: NEO4J_URI, NEO4J_USERNAME, NEO4J_DATABASE, LLM_PROVIDER, GEMINI_MODEL_NAME, ENABLE_LLM_SYNTHESIS
```

### Kubernetes Networking

Inside Kubernetes, the frontend does not call `localhost:8000`. Browser requests go to the frontend origin as `/api/...`; Nginx receives them and proxies to:

```text
http://backend-service:8000/
```

`backend-service` is resolved by Kubernetes internal DNS. This is why the frontend can use the same relative `/api` path in the browser while the container reaches the backend through the cluster service.

### Deployment Files Added To Project Structure

The documentation project tree should now be read as including these deployment files:

```text
C:\Projects\PFA
|-- Dockerfile
|-- k8s
|   |-- backend-deployment.yaml
|   `-- frontend-deployment.yaml
`-- frontend
    |-- Dockerfile
    `-- nginx.conf
```

## 31. Summary

This project implements a complete academic Agentic Graph-Vector RAG system:

- local corpus loading
- chunking evaluation
- vector embedding and FAISS index creation
- retrieval evaluation
- graph extraction through Colab
- Neo4j Aura graph store
- Louvain community detection
- React graph visualization
- rule-based query classification
- Q-learning route adaptation
- graph/vector hybrid evidence fusion
- LLM final response generation
- persistent response caching
- user feedback loop
- query validation for domain, coherence, and clarity
- Docker containerization for backend and frontend
- Nginx frontend serving and `/api` reverse proxying
- Kubernetes Deployments and NodePort Services for local cluster deployment

The system is designed to demonstrate how an agent can dynamically choose between vector search and graph search, use reinforcement learning to improve routing decisions, and produce a final natural-language answer grounded in retrieved evidence.
