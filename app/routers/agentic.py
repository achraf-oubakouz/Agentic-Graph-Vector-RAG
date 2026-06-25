from fastapi import APIRouter, HTTPException

from app.schemas.models import (
    AgenticAskRequest,
    AgenticAskResponse,
    AgenticFeedbackRequest,
    AgenticFeedbackResponse,
)
from app.services.agentic_service import ask_agent, debug_agent_evidence, get_policy, save_feedback

router = APIRouter(prefix="/agentic", tags=["Agentic Graph-Vector RAG"])


@router.post("/ask", response_model=AgenticAskResponse)
def ask(payload: AgenticAskRequest) -> dict:
    try:
        return ask_agent(payload.query, limit=payload.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/debug")
def debug(payload: AgenticAskRequest) -> dict:
    try:
        return debug_agent_evidence(payload.query, limit=payload.limit)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/feedback", response_model=AgenticFeedbackResponse)
def feedback(payload: AgenticFeedbackRequest) -> dict:
    try:
        return save_feedback(payload.query, payload.route, payload.reward)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/policy")
def policy() -> dict:
    return get_policy()
