import json
import hashlib
import re
from pathlib import Path

from app.core.config import settings
from app.core.database import store
from app.routers.query import classify_query
from app.services.embeddings_service import build_embeddings
from app.services.graph_service import graph_answer
from app.services.llm_service import (
    _build_prompt,
    _format_control_factors,
    _format_graph_evidence,
    _format_related_entities,
    _format_vector_evidence,
    _is_french,
    synthesize_answer,
)
from app.services.retrieval_service import run_retrieval
from app.services.rl_service import choose_route, policy_status, update_policy

AGENTIC_VECTOR_METHOD = "4.Hybrid BM25+Emb"
CACHE_VERSION = "agentic-cache-v7-response-quality"
CONTROL_QUERY_EXPANSION = (
    " surface heat fluxes wind stress freshwater forcing east-west SST gradient "
    "trade winds mean-state biases equatorial warming Southeast Pacific cooling"
)


DOMAIN_TERMS = {
    "pacifique", "pacific", "tropical", "climat", "climate", "climatique",
    "rechauffement", "warming", "sst", "temperature", "ocean", "oceanique",
    "atmosphere", "equateur", "equatorial", "enso", "walker", "bjerknes",
    "cmip", "modele", "modeles", "model", "models", "projection",
    "projections", "biais", "bias", "flux", "chaleur", "heat", "surface",
    "louvain", "communaute", "communautes", "community", "communities",
    "graphe", "graph", "neo4j", "faiss", "rag", "vector", "vecteur",
    "relation", "relations", "entite", "entites", "entity", "entities",
}

CLIMATE_CONCEPT_TERMS = {
    "walker", "circulation", "sst", "temperature", "gradient", "cold tongue",
    "trade winds", "wind", "winds", "cmip", "model", "models", "projection",
    "projections", "pacific", "tropical", "warming", "enso", "bjerknes",
    "flux", "heat", "ocean", "atmosphere",
}

IMPOSSIBLE_PROPERTY_TERMS = {
    "taste", "tastes", "smell", "smells", "louder", "loud", "quiet",
    "jealous", "hungry", "dream", "dreams", "angry", "happy", "sad",
    "speak", "speaks", "talk", "talks", "listen", "listens", "feel",
    "feels", "emotion", "emotions", "intention", "intentions",
}

QUESTION_TERMS = {
    "what", "why", "how", "which", "who", "where", "when",
    "explain", "describe", "show", "list",
    "quel", "quelle", "quels", "quelles", "pourquoi", "comment",
    "explique", "decris", "liste",
}

VERB_TERMS = {
    "is", "are", "was", "were", "be", "being", "been", "do", "does", "did",
    "has", "have", "had", "control", "controls", "affect", "affects",
    "influence", "influences", "explain", "explains", "show", "shows",
    "change", "changes", "weaken", "weakens", "strengthen", "strengthens",
    "favor", "favors", "favours", "favour", "promote", "promotes",
    "drive", "drives", "shape", "shapes",
    "est", "sont", "etre", "controle", "influence", "affecte", "explique",
}


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().strip().split())


def _active_llm_model_name() -> str:
    return settings.GEMINI_MODEL_NAME if settings.LLM_PROVIDER.lower() == "gemini" else settings.LLM_MODEL_NAME


def _query_tokens(query: str) -> list[str]:
    return re.findall(r"\b[^\W\d_]+\b", _normalize_query(query), flags=re.UNICODE)


def _is_semantically_incoherent(query: str) -> bool:
    normalized = _normalize_query(query)
    has_climate_concept = any(term in normalized for term in CLIMATE_CONCEPT_TERMS)
    has_impossible_property = any(term in normalized for term in IMPOSSIBLE_PROPERTY_TERMS)
    return has_climate_concept and has_impossible_property


def _is_unclear_or_ungrammatical(query: str) -> bool:
    normalized = _normalize_query(query)
    tokens = _query_tokens(query)
    if len(tokens) < 4:
        return True

    malformed_patterns = [
        r"\band\s+no\b",
        r"\bor\s+no\b",
        r"\bsmall\s+and\s+no\b",
        r"\bno\s+during\b",
        r"\bdo\s+the\s+when\b",
        r"\bdoes\s+the\s+when\b",
        r"\bdid\s+the\s+when\b",
        r"\bwhy\s+is\s+.+\s+and\s+no\b",
        r"\bwhat\s+.+\s+and\s+no\b",
    ]
    if any(re.search(pattern, normalized) for pattern in malformed_patterns):
        return True

    if tokens[-1] in {"and", "or", "but", "with", "during", "for", "of", "to", "et", "ou", "avec", "pendant", "de"}:
        return True

    has_question_or_command = bool(set(tokens) & QUESTION_TERMS)
    has_verb = bool(set(tokens) & VERB_TERMS)
    if has_question_or_command and not has_verb:
        return True

    return False


def _query_clarity_response(query: str) -> dict:
    policy_scores = policy_status().get("semantic", {})
    return {
        "query": query,
        "route": "Agent validation",
        "detected_type": "unclear_query",
        "policy_scores": policy_scores,
        "llm_model": _active_llm_model_name(),
        "llm_synthesis": False,
        "cached": False,
        "answer": (
            "The query contains corpus terms, but it is not grammatically or semantically clear enough to answer. "
            "Please reformulate it as a complete question."
        ),
        "vector_results": [],
        "graph_results": [],
        "communities": [],
    }


def _semantic_incoherence_response(query: str) -> dict:
    policy_scores = policy_status().get("semantic", {})
    return {
        "query": query,
        "route": "Agent validation",
        "detected_type": "semantically_incoherent",
        "policy_scores": policy_scores,
        "llm_model": _active_llm_model_name(),
        "llm_synthesis": False,
        "cached": False,
        "answer": (
            "The query uses corpus terms, but it asks for an impossible property or action. "
            "Please reformulate it as a scientifically meaningful climate question."
        ),
        "vector_results": [],
        "graph_results": [],
        "communities": [],
    }


def _cache_fingerprint() -> str:
    metadata_path = Path(settings.FAISS_METADATA_PATH)
    if metadata_path.exists():
        metadata_hash = hashlib.sha256(metadata_path.read_bytes()).hexdigest()[:16]
    else:
        metadata_hash = "missing"
    raw = f"{CACHE_VERSION}|{settings.LLM_PROVIDER}|{_active_llm_model_name()}|{AGENTIC_VECTOR_METHOD}|{metadata_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _cache_key(query: str, route: str, detected_type: str) -> str:
    raw = "|".join([_normalize_query(query), _cache_fingerprint()])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_agentic_cache() -> dict:
    path = Path(settings.AGENTIC_CACHE_PATH)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_agentic_cache(cache: dict) -> None:
    path = Path(settings.AGENTIC_CACHE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_in_project_domain(query: str) -> bool:
    normalized = (
        query.lower()
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ç", "c")
    )
    return any(term in normalized for term in DOMAIN_TERMS)


def _out_of_domain_response(query: str) -> dict:
    policy_scores = policy_status().get("semantic", {})
    return {
        "query": query,
        "route": "Out of corpus",
        "detected_type": "out_of_domain",
        "policy_scores": policy_scores,
        "llm_model": _active_llm_model_name(),
        "llm_synthesis": False,
        "cached": False,
        "answer": (
            "I cannot answer this question from the project corpus. "
            "This RAG system is limited to tropical Pacific warming, climate models, "
            "FAISS, Neo4j, graph relationships, and Louvain communities."
        ),
        "vector_results": [],
        "graph_results": [],
        "communities": [],
    }


def _ensure_vector_ready() -> None:
    if not store.chunks:
        metadata_path = Path(settings.FAISS_METADATA_PATH)
        if metadata_path.exists():
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            store.chunks = [item["text"] for item in data.get("chunks", [])]
            store.best_chunking_method = data.get("metadata", {}).get(
                "best_chunking_method",
                "loaded from FAISS metadata",
            )
        if not store.chunks:
            raise RuntimeError("Run /chunking first, or /pipeline.")
    if store.lsa is None or store.vec is None or store.svd is None:
        store.lsa, _, store.vec, store.svd, store.pca = build_embeddings(store.chunks)


def _vector_answer(query: str) -> dict:
    _ensure_vector_ready()
    retrieval_query = query
    normalized = _normalize_query(query)
    asks_for_mechanisms = any(
        term in normalized
        for term in (
            "control", "controls", "controlled", "influence", "influences",
            "affect", "affects", "cause", "causes", "why", "explain", "how",
        )
    )
    mentions_tropical_pacific_warming = (
        "tropical" in normalized
        and "pacific" in normalized
        and "warming" in normalized
    )
    if asks_for_mechanisms and mentions_tropical_pacific_warming:
        retrieval_query = f"{query} {CONTROL_QUERY_EXPANSION}"
    return run_retrieval(
        query=retrieval_query,
        chunks=store.chunks,
        lsa=store.lsa,
        vec=store.vec,
        svd=store.svd,
        best_chunking_method=store.best_chunking_method,
        preferred_method=AGENTIC_VECTOR_METHOD,
        result_text_chars=1400,
    )


def ask_agent(query: str, limit: int = 8) -> dict:
    if _is_unclear_or_ungrammatical(query):
        return _query_clarity_response(query)

    if _is_semantically_incoherent(query):
        return _semantic_incoherence_response(query)

    if not _is_in_project_domain(query):
        return _out_of_domain_response(query)

    classified = classify_query(query)
    learned_route, policy_scores = choose_route(
        classified.detected_type,
        classified.routing,
    )
    cache_key = _cache_key(query, learned_route, classified.detected_type)
    cache = _load_agentic_cache()
    if cache_key in cache:
        cached_response = cache[cache_key]
        return {
            **cached_response,
            "policy_scores": policy_scores,
            "cached": True,
        }

    vector_data = None
    graph_data = None

    if learned_route in {"Vectorial RAG", "Graph RAG", "Agentic Graph-Vector RAG"}:
        vector_data = _vector_answer(query)
    if learned_route in {"Graph RAG", "Agentic Graph-Vector RAG"}:
        graph_data = graph_answer(query, limit=limit)

    answer, used_llm = synthesize_answer(query, learned_route, vector_data, graph_data)

    response = {
        "query": query,
        "route": learned_route,
        "detected_type": classified.detected_type,
        "policy_scores": policy_scores,
        "llm_model": _active_llm_model_name(),
        "llm_synthesis": used_llm,
        "cached": False,
        "answer": answer,
        "vector_results": vector_data["top_results"] if vector_data else [],
        "graph_results": graph_data["results"] if graph_data else [],
        "communities": graph_data["communities"] if graph_data else [],
    }

    if used_llm:
        cache[cache_key] = response
        _save_agentic_cache(cache)

    return response


def debug_agent_evidence(query: str, limit: int = 8) -> dict:
    if _is_unclear_or_ungrammatical(query):
        return {**_query_clarity_response(query), "debug": "validation_failed"}

    if _is_semantically_incoherent(query):
        return {**_semantic_incoherence_response(query), "debug": "validation_failed"}

    if not _is_in_project_domain(query):
        return {**_out_of_domain_response(query), "debug": "validation_failed"}

    classified = classify_query(query)
    learned_route, policy_scores = choose_route(
        classified.detected_type,
        classified.routing,
    )

    vector_data = None
    graph_data = None
    if learned_route in {"Vectorial RAG", "Graph RAG", "Agentic Graph-Vector RAG"}:
        vector_data = _vector_answer(query)
    if learned_route in {"Graph RAG", "Agentic Graph-Vector RAG"}:
        graph_data = graph_answer(query, limit=limit)

    french = _is_french(query)
    vector_evidence = _format_vector_evidence(vector_data, french=french)
    graph_evidence = _format_graph_evidence(graph_data, french=french)
    related_entities = _format_related_entities(query, graph_data)
    control_factors = _format_control_factors(vector_evidence, graph_evidence, french)
    prompt = _build_prompt(
        query,
        learned_route,
        vector_evidence,
        graph_evidence,
        related_entities,
        control_factors,
        french,
    )

    return {
        "query": query,
        "detected_type": classified.detected_type,
        "classifier_route": classified.routing,
        "learned_route": learned_route,
        "policy_scores": policy_scores,
        "vector_method": AGENTIC_VECTOR_METHOD,
        "vector_results": vector_data["top_results"] if vector_data else [],
        "vector_resume": vector_data.get("resume") if vector_data else "",
        "graph_results": graph_data["results"] if graph_data else [],
        "communities": graph_data["communities"] if graph_data else [],
        "llm_inputs": {
            "french": french,
            "vector_evidence": vector_evidence,
            "graph_evidence": graph_evidence,
            "related_entities": related_entities,
            "control_factors": control_factors,
            "prompt": prompt,
            "prompt_chars": len(prompt),
        },
    }


def save_feedback(query: str, route: str, reward: float) -> dict:
    classified = classify_query(query)
    next_state = classified.detected_type
    return update_policy(classified.detected_type, route, reward, next_state=next_state)


def get_policy() -> dict:
    return policy_status()
