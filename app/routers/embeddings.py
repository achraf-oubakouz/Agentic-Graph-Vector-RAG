from fastapi import APIRouter, HTTPException

from app.core.database import store
from app.schemas.models import EmbeddingsRequest, EmbeddingsResponse
from app.services.embeddings_service import build_embeddings, run_embeddings

router = APIRouter(prefix="/embeddings", tags=["Embeddings"])


@router.post("", response_model=EmbeddingsResponse)
def create_embeddings(payload: EmbeddingsRequest) -> dict:
    if not store.chunks:
        raise HTTPException(
            status_code=400,
            detail="Run /chunking before /embeddings.",
        )

    store.lsa, _, store.vec, store.svd, store.pca = build_embeddings(store.chunks)
    return run_embeddings(
        chunks=store.chunks,
        query=payload.query,
        lsa=store.lsa,
        vec=store.vec,
        svd=store.svd,
        pca=store.pca,
    )
