from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import store
from app.core.security import add_cors
from app.routers import agentic, chunking, colab, corpus, embeddings, graph, query, retrieval
from app.services.chunking_service import run_chunking
from app.services.embeddings_service import build_embeddings
from app.services.retrieval_service import run_retrieval


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Agentic Graph-Vector RAG API with a FAISS vector store.",
)

add_cors(app)
app.mount("/static", StaticFiles(directory=settings.STATIC_DIR), name="static")

app.include_router(corpus.router)
app.include_router(chunking.router)
app.include_router(embeddings.router)
app.include_router(retrieval.router)
app.include_router(query.router)
app.include_router(colab.router)
app.include_router(graph.router)
app.include_router(agentic.router)


@app.on_event("startup")
def startup() -> None:
    store.load_corpus()


@app.get("/")
def root() -> dict:
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "corpus_loaded": bool(store.text),
        "vector_store": "FAISS",
    }


@app.post("/pipeline")
def run_full_pipeline(query_text: str = "What are the main topics discussed in this document?") -> dict:
    if not store.text.strip():
        raise HTTPException(status_code=400, detail="No corpus loaded.")

    chunking_result = run_chunking(store.text)
    store.chunks = chunking_result["best_chunks"]
    store.best_chunking_method = chunking_result["best_method"]

    store.lsa, _, store.vec, store.svd, store.pca = build_embeddings(store.chunks)

    retrieval_result = run_retrieval(
        query=query_text,
        chunks=store.chunks,
        lsa=store.lsa,
        vec=store.vec,
        svd=store.svd,
        best_chunking_method=store.best_chunking_method,
    )

    return {
        "query": query_text,
        "chunking": {
            "best_method": chunking_result["best_method"],
            "num_chunks": chunking_result["num_chunks"],
            "comparison_image": chunking_result["comparison_image"],
            "hierarchy_image": chunking_result["hierarchy_image"],
        },
        "retrieval": retrieval_result,
    }
