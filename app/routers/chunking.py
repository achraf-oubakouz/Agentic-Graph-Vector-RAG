from fastapi import APIRouter, HTTPException

from app.core.database import store
from app.schemas.models import ChunkingResponse
from app.services.chunking_service import run_chunking

router = APIRouter(prefix="/chunking", tags=["Chunking"])


@router.post("", response_model=ChunkingResponse)
def chunk_corpus() -> dict:
    if not store.text.strip():
        raise HTTPException(status_code=400, detail="No corpus loaded.")

    result = run_chunking(store.text)
    store.chunks = result["best_chunks"]
    store.best_chunking_method = result["best_method"]

    return {
        "best_method": result["best_method"],
        "num_chunks": result["num_chunks"],
        "all_metrics": result["all_metrics"],
        "comparison_image": result["comparison_image"],
        "hierarchy_image": result["hierarchy_image"],
    }
