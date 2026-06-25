from fastapi import APIRouter, HTTPException

from app.core.database import store
from app.schemas.models import RetrievalRequest, RetrievalResponse
from app.services.embeddings_service import build_embeddings
from app.services.retrieval_service import run_retrieval

router = APIRouter(prefix="/retrieval", tags=["Retrieval"])


@router.post("", response_model=RetrievalResponse)
def retrieve(payload: RetrievalRequest) -> dict:
    if not store.chunks:
        raise HTTPException(
            status_code=400,
            detail="Run /chunking before /retrieval.",
        )

    if store.lsa is None or store.vec is None or store.svd is None:
        store.lsa, _, store.vec, store.svd, store.pca = build_embeddings(store.chunks)

    return run_retrieval(
        query=payload.query,
        chunks=store.chunks,
        lsa=store.lsa,
        vec=store.vec,
        svd=store.svd,
        best_chunking_method=store.best_chunking_method,
    )
