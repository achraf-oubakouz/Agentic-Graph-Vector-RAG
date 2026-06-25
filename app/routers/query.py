from fastapi import APIRouter
import re

from app.core.database import store
from app.schemas.models import HistoryResponse, QueryRequest, QueryResponse

router = APIRouter(prefix="/query-router", tags=["Query Router"])

RELATIONAL_TERMS = {
    "who",
    "which",
    "relation",
    "related",
    "connect",
    "linked",
    "between",
    "project",
    "department",
    "works",
    "collaborates",
    "qui",
    "quel",
    "quels",
    "relation",
    "relie",
    "liens",
    "control",
    "controls",
    "influence",
    "influences",
    "affect",
    "affects",
    "cause",
    "causes",
    "depend",
    "depends",
    "contribute",
    "contributes",
    "measure",
    "measures",
    "correct",
    "corrects",
    "compare",
    "compares",
    "contrôle",
    "controle",
    "influence",
    "affecte",
    "cause",
    "dépend",
    "depend",
    "contribue",
}

CONCEPTUAL_TERMS = {
    "what",
    "explain",
    "define",
    "describe",
    "summary",
    "why",
    "how",
    "quoi",
    "explique",
    "definition",
    "résume",
    "resume",
}

RELATIONAL_PHRASES = {
    "depends on",
    "based on",
    "part of",
    "linked to",
    "related to",
    "connected to",
    "contributes to",
    "in relation to",
    "depend de",
    "dépend de",
    "lié à",
    "lie a",
    "relie à",
    "relie a",
}

CONCEPTUAL_ONLY_TERMS = {
    "define",
    "definition",
    "summary",
    "resume",
    "résume",
    "describe",
}

GENERIC_QUESTION_TERMS = {
    "what",
    "which",
    "who",
    "quoi",
    "qui",
    "quel",
    "quels",
    "quelle",
    "quelles",
}


def classify_query(query: str) -> QueryResponse:
    normalized = query.lower()
    tokens = set(re.findall(r"[a-zA-ZÀ-ÿ0-9]+", normalized))
    graph_hits = (tokens & RELATIONAL_TERMS) - GENERIC_QUESTION_TERMS
    vector_hits = tokens & CONCEPTUAL_TERMS
    phrase_hits = {phrase for phrase in RELATIONAL_PHRASES if phrase in normalized}

    graph_score = (2 * len(graph_hits)) + (3 * len(phrase_hits))
    vector_score = len(vector_hits)

    # Relation verbs are stronger than generic question words such as "what".
    if graph_score >= 2 and (vector_hits - GENERIC_QUESTION_TERMS):
        detected_type = "hybrid"
        routing = "Agentic Graph-Vector RAG"
        confidence = 0.85
    elif graph_score >= 2:
        detected_type = "systematic"
        routing = "Graph RAG"
        confidence = 0.8

    elif graph_score and vector_score:
        detected_type = "hybrid"
        routing = "Agentic Graph-Vector RAG"
        confidence = 0.85
    elif graph_score > vector_score:
        detected_type = "systematic"
        routing = "Graph RAG"
        confidence = 0.75
    else:
        detected_type = "semantic"
        routing = "Vectorial RAG"
        confidence = 0.75 if vector_score else 0.55

    return QueryResponse(
        query=query,
        detected_type=detected_type,
        routing=routing,
        confidence=confidence,
    )


@router.post("", response_model=QueryResponse)
def route_query(payload: QueryRequest) -> QueryResponse:
    result = classify_query(payload.query)
    store.query_history.append(result.model_dump())
    return result


@router.get("/history", response_model=HistoryResponse)
def query_history() -> dict:
    return {"queries": store.query_history}
