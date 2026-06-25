from functools import lru_cache
import re
import unicodedata

from app.core.config import settings


FRENCH_HINTS = {
    "le", "la", "les", "des", "du", "une", "dans", "avec", "pour",
    "quel", "quelle", "quels", "quelles", "explique", "resume",
    "rechauffement", "pacifique", "modele", "modeles", "climatique",
    "biais", "equateur", "ocean", "these", "reponse", "projection",
    "temperature", "communaute", "communautes", "relation", "relations",
}

ENGLISH_HINTS = {
    "the", "and", "with", "for", "warming", "climate", "models",
    "model", "response", "projection", "evidence", "this", "that",
    "eastern", "western", "surface", "temperature",
}

RELATION_FR = {
    "RELATED_TO": "est associe a",
    "CO_OCCURS_WITH": "apparait avec",
    "LOCATED_IN": "est situe dans",
    "CAUSES": "cause",
    "INFLUENCES": "influence",
    "CONTROLS": "controle",
    "AFFECTS": "affecte",
    "DEPENDS_ON": "depend de",
    "CONTRIBUTES_TO": "contribue a",
    "USES": "utilise",
    "BASED_ON": "repose sur",
    "PART_OF": "fait partie de",
    "PROJECTS": "projette",
    "CORRECTS": "corrige",
    "EXPLAINS": "explique",
    "MEASURES": "mesure",
    "COMPARES_WITH": "compare avec",
}

GENERIC_GRAPH_NODES = {
    "figure", "table", "les", "cette", "ce", "cet", "il", "elle", "nous",
    "article", "chapter", "section", "the", "this", "for", "while", "first",
    "keywords", "source", "equation", "climatologie", "climatology",
}


def _word_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"\b[^\W\d_]+\b", str(text), flags=re.UNICODE)
        if token
    ]


def _is_entity_question(query: str) -> bool:
    normalized = _normalize_for_language(query)
    return bool(
        re.search(
            r"\b(list|show|entities|entity|connected|related|relationships|"
            r"quelles|quels|entites|entite|relations|relie|connecte)\b",
            normalized,
        )
        or re.search(r"\b(which|what)\s+(entities|relationships|relations)\b", normalized)
    )


def _is_control_question(query: str) -> bool:
    normalized = _normalize_for_language(query)
    return bool(
        re.search(
            r"\b(control|controls|controlled|influence|influences|affect|affects|cause|causes|"
            r"shape|shapes|shaping|why|explain|controle|influence|affecte|cause|pourquoi|explique)\b",
            normalized,
        )
    )


def _is_results_question(query: str) -> bool:
    normalized = _normalize_for_language(query)
    return bool(
        re.search(
            r"\b(result|results|finding|findings|conclusion|conclusions|outcome|outcomes|"
            r"show|shows|found|reveals|revealed|main result|main findings|"
            r"resultat|resultats|conclusion|conclusions|montre|montrent)\b",
            normalized,
        )
    )


def _is_title_style_query(query: str) -> bool:
    tokens = _word_tokens(query)
    if not tokens:
        return False
    uppercaseish = sum(1 for token in tokens if token[:1].isupper())
    return query.lower().startswith(("explain ", "describe ", "summarize ")) and uppercaseish >= max(3, len(tokens) // 2)


def _normalize_for_language(text: str) -> str:
    lowered = str(text).lower()
    lowered = "".join(
        char
        for char in unicodedata.normalize("NFKD", lowered)
        if not unicodedata.combining(char)
    )
    return (
        lowered
        .replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("ë", "e")
        .replace("à", "a")
        .replace("â", "a")
        .replace("î", "i")
        .replace("ï", "i")
        .replace("ô", "o")
        .replace("ù", "u")
        .replace("û", "u")
        .replace("ç", "c")
        .replace("œ", "oe")
    )


def _is_french(query: str) -> bool:
    lowered = _normalize_for_language(query)
    if re.search(r"[àâçéèêëîïôùûüÿœ]", query.lower()):
        return True
    return bool(
        re.search(
            r"\b(qui|que|quel|quelle|quels|quelles|explique|resume|dans|lie|lies|"
            r"controle|rechauffement|pacifique|communaute|communautes|entites)\b",
            lowered,
        )
    )


def _language_score(text: str, hints: set[str]) -> int:
    lowered = _normalize_for_language(text)
    return sum(1 for hint in hints if hint in lowered)


def _clean_text(text: str) -> str:
    text = " ".join(str(text).split())
    for marker in ["Acronym", "Notation", "Definition", "Usage", "FIGURE", "TABLE", "---"]:
        if marker in text:
            text = text.split(marker, 1)[0]
    return text.strip(" .;:-")


def _sentences(text: str) -> list[str]:
    cleaned = _clean_text(text)
    return [
        sentence.strip(" .;:-")
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if 45 <= len(sentence.strip()) <= 420
    ]


def _format_vector_evidence(vector_data: dict | None, french: bool, query: str = "") -> str:
    if not vector_data:
        return ""

    candidates = []
    result_question = _is_results_question(query)
    query_terms = {
        token
        for token in _word_tokens(_normalize_for_language(query))
        if len(token) > 4 and token not in {"result", "results", "study", "climate", "models"}
    }
    result_markers = {
        "driven", "controlled", "explained", "reveals", "show", "shows",
        "primarily", "predominantly", "linked", "complicating", "reduction",
        "cooling", "warming", "diversity", "spread", "bias", "biases",
        "projette", "montre", "explique", "controle", "refroidissement",
        "rechauffement", "diversite", "biais",
    }
    aim_markers = {
        "aim", "aims", "objective", "objectives", "proposes", "method",
        "methodological", "framework", "assessing", "using dedicated",
        "objectif", "objectifs", "methode", "cadre",
    }

    for item in vector_data.get("top_results", [])[:8]:
        for sentence in _sentences(item.get("text", "")):
            fr_score = _language_score(sentence, FRENCH_HINTS)
            en_score = _language_score(sentence, ENGLISH_HINTS)
            if french and en_score > fr_score + 1:
                continue
            if not french and fr_score > en_score + 1:
                continue
            score = fr_score if french else en_score
            normalized_sentence = _normalize_for_language(sentence)
            sentence_terms = set(_word_tokens(normalized_sentence))
            score += len(query_terms & sentence_terms) * 2
            if result_question:
                score += len(result_markers & sentence_terms) * 3
                if any(marker in normalized_sentence for marker in aim_markers):
                    score -= 5
            candidates.append((score, sentence))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    for _, sentence in candidates:
        key = _normalize_for_language(sentence)[:90]
        if key in seen:
            continue
        seen.add(key)
        selected.append(sentence)
        if len(selected) == 4:
            break

    return "\n".join(f"- {sentence}" for sentence in selected)


def _format_graph_evidence(graph_data: dict | None, french: bool) -> str:
    if not graph_data:
        return ""
    lines = []
    for item in graph_data.get("results", [])[:8]:
        source = str(item["source"]).strip()
        target = str(item["target"]).strip()
        if _normalize_for_language(source) in GENERIC_GRAPH_NODES:
            continue
        if _normalize_for_language(target) in GENERIC_GRAPH_NODES:
            continue
        relation = item["relationship"]
        if relation == "CO_OCCURS_WITH":
            continue
        if french:
            relation = RELATION_FR.get(relation, relation.lower().replace("_", " "))
        lines.append(f"- {source} {relation} {target}")
    return "\n".join(lines)


def _format_related_entities(query: str, graph_data: dict | None) -> str:
    if not graph_data:
        return ""
    query_terms = {
        token
        for token in _word_tokens(_normalize_for_language(query))
        if len(token) > 3
    }
    entities = []
    for item in graph_data.get("results", [])[:20]:
        source = str(item["source"]).strip()
        target = str(item["target"]).strip()
        normalized_source = _normalize_for_language(source)
        normalized_target = _normalize_for_language(target)
        if normalized_source in GENERIC_GRAPH_NODES or normalized_target in GENERIC_GRAPH_NODES:
            continue
        source_matches = any(term in normalized_source for term in query_terms)
        target_matches = any(term in normalized_target for term in query_terms)
        if source_matches and not target_matches:
            entities.append(target)
        elif target_matches and not source_matches:
            entities.append(source)
        else:
            entities.extend([source, target])

    unique_entities = []
    seen = set()
    for entity in entities:
        key = _normalize_for_language(entity)
        if key in seen:
            continue
        seen.add(key)
        unique_entities.append(entity)
        if len(unique_entities) == 12:
            break

    return ", ".join(unique_entities)


def _format_control_factors(vector_evidence: str, graph_evidence: str, french: bool) -> str:
    evidence = _normalize_for_language(f"{vector_evidence}\n{graph_evidence}")
    factors = []
    candidates = [
        ("sea-surface temperature patterns", "les structures de temperature de surface de la mer", ("sst", "sea surface temperature", "temperature de surface")),
        ("the Walker circulation", "la circulation de Walker", ("walker",)),
        ("Bjerknes feedback", "la retroaction de Bjerknes", ("bjerknes",)),
        ("the east-west ocean-atmosphere gradient", "le gradient est-ouest ocean-atmosphere", ("east-west", "est-ouest", "gradient")),
        ("equatorial ocean dynamics", "la dynamique oceanique equatoriale", ("equatorial", "equateur", "equatoriale")),
        ("surface heat fluxes", "les flux de chaleur de surface", ("heat flux", "flux", "evapor", "evaporation")),
        ("mean-state biases in climate models", "les biais d'etat moyen des modeles climatiques", ("mean state", "etat moyen", "bias", "biais")),
    ]
    for english, french_label, terms in candidates:
        if any(term in evidence for term in terms):
            factors.append(french_label if french else english)

    unique_factors = []
    for factor in factors:
        if factor not in unique_factors:
            unique_factors.append(factor)
    return ", ".join(unique_factors[:6])


def _has_bad_repetition(answer: str) -> bool:
    lowered = _normalize_for_language(answer)
    words = lowered.split()
    if len(words) > 35 and len(set(words)) / len(words) < 0.35:
        return True
    phrases = re.findall(r"\b[\wÀ-ÿ]+(?:\s+[\wÀ-ÿ]+){1,4}\b", lowered)
    counts = {}
    for phrase in phrases:
        counts[phrase] = counts.get(phrase, 0) + 1
    return bool(counts and max(counts.values()) >= 4)


def _has_mixed_language(answer: str, french: bool) -> bool:
    if not french:
        return False
    english_markers = {"the", "this", "through", "with", "warming", "models", "evidence"}
    words = set(re.findall(r"[A-Za-z]+", answer.lower()))
    return len(words & english_markers) >= 2


def _leaks_prompt_labels(answer: str) -> bool:
    lowered = _normalize_for_language(answer)
    labels = {
        "graph rag", "vector rag", "agentic graph-vector rag",
        "preuves du graphe", "preuves vectorielles", "graph evidence",
        "vector evidence", "route utilisee", "route used", "reponse:",
        "answer:", "french answer", "selected route", "route",
        "required model", "question:",
    }
    return any(label in lowered for label in labels)


def _is_too_weak_answer(answer: str, query: str) -> bool:
    cleaned = _clean_text(answer)
    normalized_answer = _normalize_for_language(cleaned).strip(" ?.!")
    normalized_query = _normalize_for_language(query).strip(" ?.!")
    if normalized_answer == normalized_query:
        return True
    if normalized_answer.startswith(normalized_query[: max(18, min(len(normalized_query), 60))]):
        return True
    words = _word_tokens(cleaned)
    if _is_entity_question(query):
        return len(words) < 2
    if len(words) < 8:
        return True
    if _is_results_question(query):
        finding_terms = {
            "result", "results", "finding", "findings", "show", "shows",
            "found", "reveals", "revealed", "driven", "controlled",
            "explained", "primarily", "predominantly", "linked", "diversity",
            "spread", "cooling", "warming", "bias", "biases", "wind",
            "winds", "cloud", "clouds", "ocean", "dynamics", "gradient",
            "resultat", "resultats", "montre", "montrent", "explique",
            "controle", "diversite", "biais",
        }
        answer_terms = set(_word_tokens(normalized_answer))
        aim_only = re.search(
            r"\b(aim|aims|objective|objectives|proposes|framework|method|"
            r"objectif|objectifs|methode|cadre)\b",
            normalized_answer,
        )
        if aim_only and len(answer_terms & finding_terms) < 2:
            return True
        if len(words) >= 18 and len(answer_terms & finding_terms) >= 2:
            return False
    if _is_control_question(query):
        mechanism_terms = {
            "sst", "sea", "surface", "temperature", "walker", "bjerknes",
            "gradient", "bias", "biases", "flux", "fluxes", "heat", "circulation",
            "wind", "stress",
            "equatorial", "dynamics", "evaporation", "evaporative",
            "etat", "moyen", "biais", "circulation", "equatoriale",
        }
        answer_terms = set(_word_tokens(_normalize_for_language(cleaned)))
        if not answer_terms & mechanism_terms:
            return True
        generic_only = (
            "reliability of these projections" in _normalize_for_language(cleaned)
            or "understanding the mechanisms" in _normalize_for_language(cleaned)
        )
        if generic_only and len(answer_terms & mechanism_terms) < 2:
            return True
        if len(words) >= 18 and len(answer_terms & mechanism_terms) >= 2:
            return False
    if cleaned.lower().startswith(("ound ", "ject ", "odel ")):
        return True
    if not re.search(r"[.!?]$", cleaned):
        return True
    if re.search(r"\b(proc|meth|mod|sim|contrib|divers|analys|expl)\.?$", cleaned.lower()):
        return True
    if len(words) >= 35:
        return False
    if _is_title_style_query(query):
        normalized = _normalize_for_language(cleaned)
        if "using" in normalized and len(words) < 35:
            return True
    query_terms = {
        word
        for word in _word_tokens(_normalize_for_language(query))
        if len(word) > 4
    }
    answer_terms = set(_word_tokens(_normalize_for_language(answer)))
    return bool(query_terms) and not bool(query_terms & answer_terms)


def _copies_evidence_fragment(answer: str, vector_evidence: str, graph_evidence: str) -> bool:
    normalized_answer = _normalize_for_language(answer)
    evidence = _normalize_for_language(f"{vector_evidence}\n{graph_evidence}")
    answer_words = _word_tokens(normalized_answer)
    if len(answer_words) < 16:
        return False
    for size in (18, 16, 14):
        for index in range(0, max(len(answer_words) - size + 1, 0)):
            phrase = " ".join(answer_words[index:index + size])
            if len(phrase) > 90 and phrase in evidence:
                return True
    return False


def _cleanup_generated_answer(answer: str) -> str:
    cleaned = " ".join(str(answer).split())
    for marker in (
        "Final answer:",
        "Final Answer:",
        "Answer:",
        "Reponse finale:",
        "Réponse finale:",
        "Reponse:",
        "Réponse:",
    ):
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
    cleaned = re.sub(r"^\s*[-:;,\.\s]+", "", cleaned).strip()
    if cleaned and not re.search(r"[.!?]$", cleaned) and len(_word_tokens(cleaned)) >= 10:
        cleaned = f"{cleaned}."
    return cleaned


def _has_severe_generation_problem(answer: str, query: str, french: bool) -> bool:
    if not answer:
        return True
    if len(_word_tokens(answer)) < 8:
        return True
    normalized_answer = _normalize_for_language(_clean_text(answer)).strip(" ?.!")
    normalized_query = _normalize_for_language(query).strip(" ?.!")
    if normalized_answer == normalized_query:
        return True
    return (
        _has_bad_repetition(answer)
        or _has_mixed_language(answer, french)
        or _leaks_prompt_labels(answer)
    )


def _summarize_vector_french(vector: str) -> str:
    normalized = _normalize_for_language(vector)
    concepts = []
    if "walker" in normalized:
        concepts.append("la circulation de Walker")
    if "bjerknes" in normalized:
        concepts.append("la retroaction de Bjerknes")
    if "gradient est-ouest" in normalized or "gradient" in normalized:
        concepts.append("le gradient est-ouest ocean-atmosphere")
    if "biais" in normalized and "etat moyen" in normalized:
        concepts.append("les biais d'etat moyen des modeles")
    if "evapor" in normalized or "flux" in normalized or "refroidissement" in normalized:
        concepts.append("les flux de chaleur et le refroidissement relatif")
    if "equateur" in normalized or "equatorial" in normalized:
        concepts.append("la dynamique equatoriale")

    unique_concepts = []
    for concept in concepts:
        if concept not in unique_concepts:
            unique_concepts.append(concept)

    if unique_concepts:
        joined = ", ".join(unique_concepts[:-1])
        if len(unique_concepts) > 1:
            joined = f"{joined} et {unique_concepts[-1]}"
        else:
            joined = unique_concepts[0]
        return (
            "Le contexte vectoriel indique que le rechauffement du Pacifique "
            f"Tropical est principalement explique par {joined}."
        )

    first_sentence = vector.splitlines()[0].lstrip("- ").strip()
    first_sentence = re.sub(r"^(resume|résumé)\s+", "", first_sentence, flags=re.IGNORECASE)
    return f"Le contexte vectoriel indique que {first_sentence[0].lower() + first_sentence[1:]}."


def _summarize_vector_english(vector: str) -> str:
    normalized = _normalize_for_language(vector)
    concepts = []
    if "walker" in normalized:
        concepts.append("changes in the Walker circulation")
    if "bjerknes" in normalized:
        concepts.append("Bjerknes feedback")
    if "east-west gradient" in normalized or "gradient" in normalized:
        concepts.append("the east-west ocean-atmosphere gradient")
    if "mean state" in normalized or "bias" in normalized:
        concepts.append("mean-state biases in climate models")
    if "evapor" in normalized or "flux" in normalized or "cooling" in normalized:
        concepts.append("surface heat fluxes and evaporative cooling")
    if "equator" in normalized or "equatorial" in normalized:
        concepts.append("equatorial ocean dynamics")
    if "sst" in normalized or "sea surface temperature" in normalized:
        concepts.append("sea-surface temperature changes")

    unique_concepts = []
    for concept in concepts:
        if concept not in unique_concepts:
            unique_concepts.append(concept)

    if unique_concepts:
        if len(unique_concepts) == 1:
            joined = unique_concepts[0]
        else:
            joined = ", ".join(unique_concepts[:-1]) + f", and {unique_concepts[-1]}"
        return (
            "The retrieved evidence indicates that Pacific warming is shaped by "
            f"{joined}. These processes affect how heat is distributed across the "
            "tropical Pacific and help explain why climate-model projections differ."
        )

    first_sentence = vector.splitlines()[0].lstrip("- ").strip() if vector else ""
    if first_sentence:
        return f"The retrieved evidence indicates that {first_sentence[0].lower() + first_sentence[1:]}."
    return ""


def _fallback_answer(query: str, vector_data: dict | None, graph_data: dict | None) -> str:
    french = _is_french(query)
    vector = _format_vector_evidence(vector_data, french=french, query=query)
    graph = _format_graph_evidence(graph_data, french=french)

    if french:
        parts = []
        if graph:
            first_relations = "; ".join(line[2:] for line in graph.splitlines()[:3])
            parts.append(f"Le graphe met en evidence ces relations: {first_relations}.")
        if vector:
            parts.append(_summarize_vector_french(vector))
        return " ".join(parts) if parts else "Aucune evidence suffisante n'a ete retrouvee pour repondre clairement."

    parts = []
    if graph:
        first_relations = "; ".join(line[2:] for line in graph.splitlines()[:3])
        parts.append(f"The graph highlights these relationships: {first_relations}.")
    if vector:
        parts.append(_summarize_vector_english(vector))
    return " ".join(parts) if parts else "Not enough evidence was retrieved to answer clearly."


def _active_model_name() -> str:
    if settings.LLM_PROVIDER.lower() == "gemini":
        return settings.GEMINI_MODEL_NAME
    return settings.LLM_MODEL_NAME


@lru_cache(maxsize=1)
def _load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_NAME)
    model_kwargs = {"torch_dtype": torch.float16 if torch.cuda.is_available() else "auto"}
    model = AutoModelForCausalLM.from_pretrained(settings.LLM_MODEL_NAME, **model_kwargs)
    if torch.cuda.is_available():
        model = model.to("cuda")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return tokenizer, model


def _format_chat_prompt(tokenizer, prompt: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return prompt
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def _build_prompt(
    query: str,
    route: str,
    vector_evidence: str,
    graph_evidence: str,
    related_entities: str,
    control_factors: str,
    french: bool,
) -> str:
    if french:
        return f"""
CONTROLE DE COHERENCE SEMANTIQUE IMPORTANT:
Ne reponds jamais litteralement a une question impossible.
Les mecanismes climatiques ne peuvent pas ressentir d'emotions, avoir un gout, parler, rever, etre jaloux, devenir bruyants ou avoir des intentions.
Si la question attribue ce type de propriete impossible a un concept climatique, reponds exactement: "La question utilise des termes du corpus, mais elle est semantiquement incoherente. Les concepts climatiques n'ont pas de proprietes sensorielles ou emotionnelles."
Si la question contient seulement un ordre de mots casse ou une grammaire peu claire, demande de la reformuler; n'utilise pas la reponse sur les emotions ou les proprietes impossibles.

Tu es un assistant RAG scientifique. Ta tache est de produire une reponse explicative, claire et pertinente a partir des preuves fournies.

Question: {query}

Preuves relationnelles issues du graphe Neo4j:
{graph_evidence or "Aucune"}

Entites candidates issues du graphe:
{related_entities or "Aucune"}

Facteurs de controle detectes dans les preuves:
{control_factors or "Aucun facteur explicite detecte"}

Passages pertinents issus du Vector Store:
{vector_evidence or "Aucune"}

Instructions:
- Reponds uniquement en francais.
- Ne repete pas la question.
- Ne mentionne pas le modele, le routage, Neo4j, FAISS, Vector Store ou Graph Store.
- Si la question demande des entites, cite les entites pertinentes et explique brievement leur lien.
- Si la question demande une explication, nomme les facteurs principaux puis explique pourquoi ils sont importants.
- Pour une question de controle ou de cause, commence par les facteurs de controle, pas par l'incertitude des projections.
- Si la question demande les resultats ou conclusions d'une etude, commence par les resultats principaux; ne reponds pas seulement avec l'objectif, la methode ou le cadre de l'etude.
- Si la question ressemble au titre d'un article ou d'une section, explique le sujet et ses resultats principaux; ne continue pas l'abstrait et ne cite pas les auteurs.
- Utilise seulement les preuves fournies.
- Si les preuves ne suffisent pas, dis-le brievement au lieu d'inventer des details.
- Ecris un seul paragraphe naturel de 2 a 4 phrases.

Reponse finale:
"""

    return f"""
IMPORTANT SEMANTIC COHERENCE CHECK:
Never answer an impossible question literally.
Climate mechanisms cannot feel emotions, taste, speak, dream, become jealous, become loud, or have intentions.
If the question assigns this kind of impossible property to a climate concept, answer exactly: "The query uses terms from the corpus, but it is semantically incoherent. Climate concepts do not have sensory or emotional properties."
If the question only has broken word order or unclear grammar, ask the user to reformulate it; do not use the sensory or emotional property response.

You are a scientific RAG assistant. Your task is to write a clear, relevant, explanatory answer using the provided evidence.

Question: {query}

Relational evidence from the graph:
{graph_evidence or "None"}

Candidate entities from the graph:
{related_entities or "None"}

Controlling factors detected in the evidence:
{control_factors or "No explicit controlling factor detected"}

Relevant passages from semantic retrieval:
{vector_evidence or "None"}

Instructions:
- Answer only in English.
- Do not repeat the question.
- Do not mention the model, route, Neo4j, FAISS, Vector Store, or Graph Store.
- If the question asks for entities, name the relevant entities and briefly explain how they are connected.
- If the question asks for an explanation, name the main factors first and then explain why they matter.
- For a control or cause question, start with the controlling factors, not with projection uncertainty.
- If the question asks for study results or findings, start with the main findings; do not answer only with the study aim, method, or framework.
- If the question looks like an article or section title, explain the topic and main findings; do not continue the abstract and do not cite authors.
- Use only the provided evidence.
- If the evidence is insufficient, say so briefly instead of inventing details.
- Write one natural paragraph of 2 to 4 sentences.

Final answer:
"""


def _build_retry_prompt(
    query: str,
    vector_evidence: str,
    graph_evidence: str,
    related_entities: str,
    control_factors: str,
    french: bool,
) -> str:
    if french:
        return f"""
CONTROLE DE COHERENCE SEMANTIQUE IMPORTANT:
Les mecanismes climatiques ne peuvent pas ressentir d'emotions, avoir un gout, parler, rever, etre jaloux, devenir bruyants ou avoir des intentions.
Si la question attribue ce type de propriete impossible a un concept climatique, reponds exactement: "La question utilise des termes du corpus, mais elle est semantiquement incoherente. Les concepts climatiques n'ont pas de proprietes sensorielles ou emotionnelles."
Si la question contient seulement un ordre de mots casse ou une grammaire peu claire, demande de la reformuler; n'utilise pas la reponse sur les emotions ou les proprietes impossibles.

Reponds en francais avec une explication courte et directe.

Question: {query}

Entites:
{related_entities or "Aucune"}

Facteurs:
{control_factors or "Aucun"}

Preuves:
{graph_evidence or vector_evidence or "Aucune preuve disponible"}

Contraintes: ne repete pas la question; pour une question sur les resultats, commence par les resultats principaux et non par l'objectif ou la methode; utilise seulement les preuves; si les preuves ne suffisent pas, dis-le brievement; donne une reponse naturelle en 2 ou 3 phrases.

Reponse:
"""

    return f"""
IMPORTANT SEMANTIC COHERENCE CHECK:
Climate mechanisms cannot feel emotions, taste, speak, dream, become jealous, become loud, or have intentions.
If the question assigns this kind of impossible property to a climate concept, answer exactly: "The query uses terms from the corpus, but it is semantically incoherent. Climate concepts do not have sensory or emotional properties."
If the question only has broken word order or unclear grammar, ask the user to reformulate it; do not use the sensory or emotional property response.

Answer in English with a short direct explanation.

Question: {query}

Entities:
{related_entities or "None"}

Factors:
{control_factors or "None"}

Evidence:
{graph_evidence or vector_evidence or "No evidence available"}

Constraints: do not repeat the question; for a results question, start with the main findings rather than the aim or method; use only the evidence; if the evidence is insufficient, say so briefly; write a natural answer in 2 or 3 sentences.

Answer:
"""


def _generate_answer(prompt: str, max_new_tokens: int | None = None) -> str:
    import torch

    tokenizer, model = _load_model()
    prompt = _format_chat_prompt(tokenizer, prompt)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=settings.LLM_MAX_INPUT_TOKENS,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens or settings.LLM_MAX_NEW_TOKENS,
            max_time=settings.LLM_GENERATION_MAX_TIME_SECONDS,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
    generated_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    return _cleanup_generated_answer(answer)


def _generate_gemini_answer(prompt: str, max_new_tokens: int | None = None) -> str:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=settings.GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.9,
            max_output_tokens=max_new_tokens or settings.GEMINI_MAX_OUTPUT_TOKENS,
            thinking_config=types.ThinkingConfig(
                thinking_budget=settings.GEMINI_THINKING_BUDGET,
            ),
        ),
    )
    finish_reason = ""
    try:
        finish_reason = str(response.candidates[0].finish_reason)
    except Exception:
        finish_reason = "unknown"
    text = getattr(response, "text", "") or ""
    if not text:
        try:
            parts = response.candidates[0].content.parts
            text = " ".join(getattr(part, "text", "") for part in parts if getattr(part, "text", ""))
        except Exception:
            text = ""
    usage = getattr(response, "usage_metadata", None)
    candidate_tokens = getattr(usage, "candidates_token_count", None) if usage else None
    thought_tokens = getattr(usage, "thoughts_token_count", None) if usage else None
    print(
        "[llm] gemini "
        f"finish_reason={finish_reason} "
        f"candidate_tokens={candidate_tokens} "
        f"thought_tokens={thought_tokens} "
        f"chars={len(text)} preview={text[:160]}"
    )
    return _cleanup_generated_answer(text)


def _generate_with_provider(prompt: str, max_new_tokens: int | None = None) -> str:
    provider = settings.LLM_PROVIDER.lower().strip()
    if provider == "gemini":
        return _generate_gemini_answer(prompt, max_new_tokens=max_new_tokens)
    if provider in {"local", "transformers", "huggingface"}:
        return _generate_answer(prompt, max_new_tokens=max_new_tokens)
    raise RuntimeError(f"Unsupported LLM_PROVIDER: {settings.LLM_PROVIDER}")


def synthesize_answer(query: str, route: str, vector_data: dict | None, graph_data: dict | None) -> tuple[str, bool]:
    french = _is_french(query)
    vector_evidence = _format_vector_evidence(vector_data, french=french, query=query)
    graph_evidence = _format_graph_evidence(graph_data, french=french)
    related_entities = _format_related_entities(query, graph_data)
    control_factors = _format_control_factors(vector_evidence, graph_evidence, french)
    unavailable_answer = (
        f"Le modele {_active_model_name()} n'a pas pu generer une reponse fiable pour cette requete."
        if french
        else f"{_active_model_name()} could not generate a reliable answer for this query."
    )

    if not settings.ENABLE_LLM_SYNTHESIS:
        return unavailable_answer, False

    prompt = _build_prompt(
        query,
        route,
        vector_evidence,
        graph_evidence,
        related_entities,
        control_factors,
        french,
    )

    try:
        answer = _generate_with_provider(prompt)
    except Exception as exc:
        print(f"[llm] generation failed: {type(exc).__name__}: {exc}")
        fallback = _fallback_answer(query, vector_data, graph_data)
        if fallback and fallback != "Not enough evidence was retrieved to answer clearly.":
            return fallback, False
        return unavailable_answer, False

    invalid = (
        _has_severe_generation_problem(answer, query, french)
        or _is_too_weak_answer(answer, query)
    )

    if invalid:
        print(f"[llm] retrying after rejected answer: {answer[:240]}")
        try:
            answer = _generate_with_provider(
                _build_retry_prompt(
                    query,
                    vector_evidence,
                    graph_evidence,
                    related_entities,
                    control_factors,
                    french,
                ),
                max_new_tokens=(
                    settings.GEMINI_RETRY_MAX_OUTPUT_TOKENS
                    if settings.LLM_PROVIDER.lower().strip() == "gemini"
                    else settings.LLM_RETRY_MAX_NEW_TOKENS
                ),
            )
        except Exception as exc:
            print(f"[llm] retry generation failed: {type(exc).__name__}: {exc}")
            fallback = _fallback_answer(query, vector_data, graph_data)
            if fallback and fallback != "Not enough evidence was retrieved to answer clearly.":
                return fallback, False
            return unavailable_answer, False

    if _has_severe_generation_problem(answer, query, french):
        print(f"[llm] rejected severe answer: {answer[:240]}")
        return unavailable_answer, False

    if _is_too_weak_answer(answer, query):
        print(f"[llm] rejected weak answer: {answer[:240]}")
        fallback = _fallback_answer(query, vector_data, graph_data)
        if fallback and fallback != "Not enough evidence was retrieved to answer clearly.":
            return fallback, False
        return unavailable_answer, False

    return answer, True
