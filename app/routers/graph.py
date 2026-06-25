from fastapi import APIRouter, HTTPException

from app.schemas.models import GraphSearchRequest, GraphSearchResponse
from app.services.graph_service import graph_network, graph_status, search_graph

router = APIRouter(prefix="/graph", tags=["Graph RAG"])


@router.get("/status")
def status() -> dict:
    try:
        return graph_status()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/search", response_model=GraphSearchResponse)
def search(payload: GraphSearchRequest) -> dict:
    try:
        return search_graph(payload.query, payload.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/network")
def network(limit: int = 1200) -> dict:
    try:
        return graph_network(limit=limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
