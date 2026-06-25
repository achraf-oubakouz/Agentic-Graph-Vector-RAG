import re
from contextlib import contextmanager

from neo4j import GraphDatabase

from app.core.config import settings


def _keywords(query: str) -> list[str]:
    stopwords = {
        "the", "and", "or", "of", "to", "in", "on", "for", "with", "what",
        "which", "who", "how", "why", "is", "are", "a", "an", "le", "la",
        "les", "de", "des", "du", "et", "ou", "dans", "qui", "quoi",
        "quel", "quels", "quelle", "quelles",
    }
    words = re.findall(r"[A-Za-zÀ-ÿ0-9]{3,}", query.lower())
    return [word for word in words if word not in stopwords][:12]


def _configured() -> bool:
    return bool(settings.NEO4J_URI and settings.NEO4J_PASSWORD)


@contextmanager
def _session():
    if not _configured():
        raise RuntimeError(
            "Neo4j Aura is not configured. Add NEO4J_URI and NEO4J_PASSWORD to .env."
        )
    driver = GraphDatabase.driver(
        settings.NEO4J_URI,
        auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
    )
    try:
        driver.verify_connectivity()
        with driver.session(database=settings.NEO4J_DATABASE) as session:
            yield session
    finally:
        driver.close()


def graph_status() -> dict:
    if not _configured():
        return {
            "configured": False,
            "connected": False,
            "message": "Missing Neo4j settings in .env",
        }

    with _session() as session:
        counts = session.run(
            """
            MATCH (e:Entity)
            OPTIONAL MATCH (e)-[r]->(:Entity)
            RETURN count(DISTINCT e) AS nodes, count(r) AS relationships
            """
        ).single()
        communities = session.run(
            """
            MATCH (e:Entity)
            WHERE e.community IS NOT NULL
            RETURN count(DISTINCT e.community) AS community_count
            """
        ).single()
        return {
            "configured": True,
            "connected": True,
            "nodes": counts["nodes"],
            "relationships": counts["relationships"],
            "community_count": communities["community_count"],
        }


def search_graph(query: str, limit: int = 25) -> dict:
    keywords = _keywords(query) or [query.lower()]

    with _session() as session:
        rows = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE any(term IN $keywords WHERE
                toLower(coalesce(a.name, '')) CONTAINS term OR
                toLower(coalesce(b.name, '')) CONTAINS term OR
                toLower(type(r)) CONTAINS term
            )
            RETURN a.name AS source,
                   type(r) AS relationship,
                   b.name AS target,
                   a.community AS source_community,
                   b.community AS target_community,
                   coalesce(r.weight, 1) AS weight
            ORDER BY weight DESC
            LIMIT $limit
            """,
            keywords=keywords,
            limit=max(1, min(limit, 100)),
        )
        results = [dict(row) for row in rows]

        communities = session.run(
            """
            MATCH (e:Entity)
            WHERE e.community IS NOT NULL
            RETURN e.community AS community,
                   count(*) AS size,
                   collect(e.name)[0..12] AS examples
            ORDER BY size DESC
            LIMIT 12
            """
        )
        community_rows = [dict(row) for row in communities]

    return {"query": query, "results": results, "communities": community_rows}


def graph_answer(query: str, limit: int = 8) -> dict:
    data = search_graph(query, limit=limit)
    if not data["results"]:
        answer = "No matching graph relationships were found in Neo4j Aura."
    else:
        lines = [
            f"{item['source']} --{item['relationship']}--> {item['target']}"
            for item in data["results"][:limit]
        ]
        answer = "Graph RAG found these relationships: " + "; ".join(lines)
    return {**data, "answer": answer}


def graph_network(limit: int = 1200) -> dict:
    with _session() as session:
        node_rows = session.run(
            """
            MATCH (e:Entity)
            RETURN elementId(e) AS id,
                   e.name AS name,
                   e.community AS community,
                   coalesce(e.mentions, 1) AS mentions
            LIMIT $limit
            """,
            limit=max(1, min(limit, 5000)),
        )
        nodes = [dict(row) for row in node_rows]
        node_ids = {node["id"] for node in nodes}

        edge_rows = session.run(
            """
            MATCH (a:Entity)-[r]->(b:Entity)
            WHERE elementId(a) IN $node_ids AND elementId(b) IN $node_ids
            RETURN elementId(a) AS source,
                   elementId(b) AS target,
                   type(r) AS relationship,
                   coalesce(r.weight, 1) AS weight
            LIMIT 5000
            """,
            node_ids=list(node_ids),
        )
        edges = [dict(row) for row in edge_rows]

    communities = {}
    for node in nodes:
        community = node.get("community")
        key = str(community) if community is not None else "unknown"
        entry = communities.setdefault(
            key,
            {"community": community, "size": 0, "mentions": 0, "examples": []},
        )
        entry["size"] += 1
        entry["mentions"] += int(node.get("mentions") or 0)
        if len(entry["examples"]) < 10:
            entry["examples"].append(node.get("name"))

    return {
        "nodes": nodes,
        "edges": edges,
        "communities": sorted(
            communities.values(),
            key=lambda item: item["size"],
            reverse=True,
        ),
    }
