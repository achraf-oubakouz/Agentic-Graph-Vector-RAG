import json
from pathlib import Path

from app.core.config import settings

ROUTES = ["Vectorial RAG", "Graph RAG", "Agentic Graph-Vector RAG"]
STATES = ["semantic", "systematic", "hybrid"]
ALPHA = 0.35
GAMMA = 0.75
POLICY_MARGIN = 0.05


def _default_policy() -> dict[str, dict[str, float]]:
    return {
        "semantic": {
            "Vectorial RAG": 0.7,
            "Graph RAG": 0.1,
            "Agentic Graph-Vector RAG": 0.3,
        },
        "systematic": {
            "Vectorial RAG": 0.1,
            "Graph RAG": 0.7,
            "Agentic Graph-Vector RAG": 0.3,
        },
        "hybrid": {
            "Vectorial RAG": 0.3,
            "Graph RAG": 0.3,
            "Agentic Graph-Vector RAG": 0.7,
        },
    }


def load_policy() -> dict[str, dict[str, float]]:
    path = Path(settings.RL_POLICY_PATH)
    if not path.exists():
        return _default_policy()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _default_policy()

    policy = _default_policy()
    for state in STATES:
        for route in ROUTES:
            if state in data and route in data[state]:
                policy[state][route] = float(data[state][route])
    return policy


def save_policy(policy: dict[str, dict[str, float]]) -> None:
    path = Path(settings.RL_POLICY_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy, indent=2), encoding="utf-8")


def choose_route(state: str, fallback_route: str) -> tuple[str, dict[str, float]]:
    policy = load_policy()
    scores = policy.get(state, {})
    if not scores:
        return fallback_route, {}
    best_route, best_score = max(scores.items(), key=lambda item: item[1])
    fallback_score = scores.get(fallback_route, 0.0)
    if best_score >= fallback_score + POLICY_MARGIN:
        return best_route, scores
    return fallback_route, scores


def update_policy(state: str, route: str, reward: float, next_state: str | None = None) -> dict:
    if state not in STATES:
        state = "hybrid"
    if next_state not in STATES:
        next_state = state
    if route not in ROUTES:
        raise ValueError(f"Unknown route: {route}")

    clipped_reward = max(-1.0, min(float(reward), 1.0))
    policy = load_policy()
    old_value = policy[state][route]
    best_next_value = max(policy[next_state].values())
    td_target = clipped_reward + GAMMA * best_next_value
    td_error = td_target - old_value
    new_value = old_value + ALPHA * td_error
    policy[state][route] = round(new_value, 4)
    save_policy(policy)
    return {
        "state": state,
        "next_state": next_state,
        "route": route,
        "reward": clipped_reward,
        "alpha": ALPHA,
        "gamma": GAMMA,
        "td_target": round(td_target, 4),
        "td_error": round(td_error, 4),
        "updated_q_value": policy[state][route],
        "policy_scores": policy[state],
    }


def policy_status() -> dict:
    return load_policy()
