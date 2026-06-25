from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.core.config import settings

router = APIRouter(prefix="/colab", tags=["Google Colab Pipeline"])


def _required_files() -> list[Path]:
    return [
        Path(settings.FAISS_INDEX_PATH),
        Path(settings.FAISS_METADATA_PATH),
        Path("colab/agentic_graph_vector_rag_colab.ipynb"),
        Path("colab/graph_rag_colab_pipeline.py"),
        Path("colab/README.md"),
    ]


@router.get("/status")
def colab_status() -> dict:
    files = _required_files()
    return {
        "ready": all(path.exists() for path in files),
        "files": [
            {
                "path": str(path),
                "exists": path.exists(),
                "size_bytes": path.stat().st_size if path.exists() else 0,
            }
            for path in files
        ],
    }


@router.post("/package")
def build_colab_package() -> FileResponse:
    missing = [str(path) for path in _required_files() if not path.exists()]
    if missing:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Run /pipeline or /retrieval first, then build the Colab package.",
                "missing": missing,
            },
        )

    package_path = Path(settings.COLAB_PACKAGE_PATH)
    package_path.parent.mkdir(parents=True, exist_ok=True)

    with ZipFile(package_path, "w", ZIP_DEFLATED) as zf:
        zf.write(settings.FAISS_INDEX_PATH, "vector_store.faiss")
        zf.write(settings.FAISS_METADATA_PATH, "vector_store_metadata.json")
        zf.write(
            "colab/agentic_graph_vector_rag_colab.ipynb",
            "agentic_graph_vector_rag_colab.ipynb",
        )
        zf.write("colab/graph_rag_colab_pipeline.py", "graph_rag_colab_pipeline.py")
        zf.write("colab/README.md", "README.md")

    return FileResponse(
        package_path,
        media_type="application/zip",
        filename=package_path.name,
    )
