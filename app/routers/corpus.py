from fastapi import APIRouter, HTTPException, UploadFile, File

from app.core.database import store
from app.schemas.models import CorpusUploadResponse

router = APIRouter(prefix="/corpus", tags=["Corpus"])


@router.get("/status")
def corpus_status() -> dict:
    return {
        "loaded": bool(store.text),
        "filename": store.filename,
        "size_chars": len(store.text),
        "size_words": len(store.text.split()),
    }


@router.post("/upload", response_model=CorpusUploadResponse)
async def upload_corpus(file: UploadFile = File(...)) -> CorpusUploadResponse:
    raw = await file.read()
    text = raw.decode("utf-8", errors="replace")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded corpus is empty.")

    store.text = text
    store.filename = file.filename or "uploaded_corpus.txt"
    store.reset()
    store.text = text
    store.filename = file.filename or "uploaded_corpus.txt"

    return CorpusUploadResponse(
        status="loaded",
        filename=store.filename,
        size_chars=len(store.text),
        size_words=len(store.text.split()),
    )
