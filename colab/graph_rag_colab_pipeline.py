import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import faiss


MODEL_NAME = os.getenv("KG_LLM_MODEL", "google/flan-t5-base")
MAX_CHUNKS = os.getenv("MAX_CHUNKS")
MAX_TRIPLES_PER_CHUNK = int(os.getenv("MAX_TRIPLES_PER_CHUNK", "10"))

RELATION_EXAMPLES = [
    "INFLUENCES",
    "CAUSES",
    "CONTROLS",
    "DEPENDS_ON",
    "CONTRIBUTES_TO",
    "AFFECTS",
    "USES",
    "BASED_ON",
    "PART_OF",
    "LOCATED_IN",
    "PROJECTS",
    "CORRECTS",
    "EXPLAINS",
    "MEASURES",
    "COMPARES_WITH",
]

RELATION_PATTERNS = [
    (r"\b(?:controls?|controle|gouverne)\b", "CONTROLS"),
    (r"\b(?:influences?|influence|modulates?|module)\b", "INFLUENCES"),
    (r"\b(?:causes?|cause|entraine|provoque)\b", "CAUSES"),
    (r"\b(?:depends? on|depend de|depends upon)\b", "DEPENDS_ON"),
    (r"\b(?:contributes? to|contribue a|participe a)\b", "CONTRIBUTES_TO"),
    (r"\b(?:affects?|affecte|impacte|modifie)\b", "AFFECTS"),
    (r"\b(?:uses?|utilise|emploie|s'appuie sur|repose sur)\b", "USES"),
    (r"\b(?:is based on|based on|base sur|fonde sur)\b", "BASED_ON"),
    (r"\b(?:is part of|part of|fait partie de)\b", "PART_OF"),
    (r"\b(?:located in|situe a|dans)\b", "LOCATED_IN"),
    (r"\b(?:projects?|projette|prevoit|annonce)\b", "PROJECTS"),
    (r"\b(?:corrects?|corrige)\b", "CORRECTS"),
    (r"\b(?:explains?|explique)\b", "EXPLAINS"),
    (r"\b(?:measures?|mesure|quantifie)\b", "MEASURES"),
    (r"\b(?:compares? with|compare a|compare)\b", "COMPARES_WITH"),
]


def clean_name(value):
    value = re.sub(r"\s+", " ", str(value)).strip(" .,:;!?()[]{}\"'")
    return value[:120]


def relation_name(value):
    value = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).upper()).strip("_")
    if not value or value in {"RELATED", "RELATED_TO", "RELATION", "ASSOCIATED"}:
        return "ASSOCIATED_WITH"
    return value[:50]


def cypher_string(value):
    return str(value).replace("\\", "\\\\").replace("'", "\\'")


def guess_label(name):
    lowered = name.lower()
    org_terms = ["university", "institute", "company", "department", "cmip", "neo4j"]
    place_terms = ["pacific", "paris", "ocean", "region", "aura"]
    if any(term in lowered for term in org_terms):
        return "Organization"
    if any(term in lowered for term in place_terms):
        return "Place"
    if len(name.split()) <= 3 and name[:1].isupper():
        return "Entity"
    return "Concept"


def extract_entities(text, limit=12):
    candidates = re.findall(
        r"\b[A-Z][A-Za-z0-9]*(?:\s+[A-Z][A-Za-z0-9]*){0,4}\b",
        text,
    )
    entities = []
    seen = set()
    stopwords = {"The", "This", "These", "Figure", "Table", "Chapter", "Abstract"}
    for item in candidates:
        item = clean_name(item)
        if len(item) < 3 or item in stopwords or item.lower() in seen:
            continue
        seen.add(item.lower())
        entities.append(item)
        if len(entities) >= limit:
            break
    return entities


def infer_relation(sentence):
    normalized = (
        sentence.replace("é", "e")
        .replace("è", "e")
        .replace("ê", "e")
        .replace("à", "a")
        .replace("î", "i")
    )
    for pattern, relation in RELATION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return relation
    return ""


def fallback_triples(text):
    triples = []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sentence in sentences:
        relation = infer_relation(sentence)
        if not relation:
            continue
        entities = extract_entities(sentence, limit=8)
        if len(entities) < 2:
            continue
        for left, right in zip(entities, entities[1:]):
            triples.append({"subject": left, "relation": relation, "object": right})
            if len(triples) >= MAX_TRIPLES_PER_CHUNK:
                return triples

    entities = extract_entities(text, limit=8)
    for left, right in zip(entities, entities[1:]):
        triples.append({"subject": left, "relation": "CO_OCCURS_WITH", "object": right})
        if len(triples) >= MAX_TRIPLES_PER_CHUNK:
            break
    return triples


def parse_json_triples(raw):
    match = re.search(r"\[[\s\S]*\]", raw)
    if not match:
        return []
    try:
        data = json.loads(match.group(0))
    except Exception:
        return []

    triples = []
    for item in data:
        if not isinstance(item, dict):
            continue
        subject = clean_name(item.get("subject", ""))
        relation = relation_name(item.get("relation", ""))
        obj = clean_name(item.get("object", ""))
        if subject and obj and subject.lower() != obj.lower():
            triples.append({"subject": subject, "relation": relation, "object": obj})
    return triples


def build_prompt(chunk):
    relation_list = ", ".join(RELATION_EXAMPLES)
    text = chunk["text"][:2200]
    return f"""
You are building a Neo4j knowledge graph from a RAG vector store.
The input below is one chunk from a FAISS vector store.

Extract concrete knowledge graph triples from the chunk.
Return only valid JSON as a list of objects.
Each object must have exactly these keys:
subject, relation, object

Rules:
- Use short canonical entity names.
- Use specific uppercase relation names.
- Do not use RELATED_TO.
- Prefer relations like: {relation_list}
- If the text says one concept affects, controls, explains, depends on, uses, corrects, measures, projects, or contributes to another, preserve that meaning.
- Maximum {MAX_TRIPLES_PER_CHUNK} triples.

Chunk id: {chunk["id"]}
Chunk text:
{text}
"""


def load_llm():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    return tokenizer, model


def generate_text(llm, prompt):
    tokenizer, model = llm
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def select_chunks(chunks):
    if MAX_CHUNKS:
        return chunks[: int(MAX_CHUNKS)]
    return chunks


def extract_triples(chunks):
    selected_chunks = select_chunks(chunks)
    try:
        llm = load_llm()
        print(f"Loaded free Hugging Face model: {MODEL_NAME}")
    except Exception as exc:
        llm = None
        print(f"LLM unavailable, using typed regex fallback: {exc}")

    triples = []
    print(f"Extracting triples from {len(selected_chunks)} chunks")
    for pos, chunk in enumerate(selected_chunks, start=1):
        chunk_triples = []
        if llm is not None:
            try:
                raw = generate_text(llm, build_prompt(chunk))
                chunk_triples = parse_json_triples(raw)
            except Exception:
                chunk_triples = []

        if not chunk_triples:
            chunk_triples = fallback_triples(chunk["text"])

        for triple in chunk_triples[:MAX_TRIPLES_PER_CHUNK]:
            triple["chunk_id"] = chunk["id"]
            triples.append(triple)

        if pos % 25 == 0:
            print(f"Processed {pos}/{len(selected_chunks)} chunks -> {len(triples)} triples")

    return triples


def build_graph(triples):
    node_counts = Counter()
    edge_counts = Counter()
    evidence = defaultdict(list)

    for triple in triples:
        s = clean_name(triple["subject"])
        r = relation_name(triple["relation"])
        o = clean_name(triple["object"])
        if not s or not o:
            continue
        node_counts[s] += 1
        node_counts[o] += 1
        edge_counts[(s, r, o)] += 1
        evidence[(s, r, o)].append(triple.get("chunk_id"))

    nodes = [
        {"id": idx, "name": name, "label": guess_label(name), "mentions": count}
        for idx, (name, count) in enumerate(node_counts.most_common())
    ]
    edges = [
        {
            "source": s,
            "relation": r,
            "target": o,
            "weight": count,
            "chunk_ids": sorted({cid for cid in evidence[(s, r, o)] if cid is not None}),
        }
        for (s, r, o), count in edge_counts.most_common()
    ]
    return nodes, edges


def apply_louvain(nodes, edges):
    try:
        import community as community_louvain
        import networkx as nx
    except Exception as exc:
        print(f"Louvain unavailable, leaving community=-1: {exc}")
        for node in nodes:
            node["community"] = -1
        return {
            "community_count": 0,
            "algorithm": "louvain_unavailable",
        }

    graph = nx.Graph()
    for node in nodes:
        graph.add_node(node["name"])
    for edge in edges:
        graph.add_edge(
            edge["source"],
            edge["target"],
            weight=max(float(edge.get("weight", 1)), 1.0),
        )

    if graph.number_of_edges() == 0:
        for node in nodes:
            node["community"] = -1
        return {
            "community_count": 0,
            "algorithm": "louvain",
        }

    partition = community_louvain.best_partition(graph, weight="weight", random_state=42)
    communities = sorted(set(partition.values()))
    for node in nodes:
        node["community"] = int(partition.get(node["name"], -1))

    print(
        "Louvain complete: "
        f"{len(communities)} communities over {graph.number_of_nodes()} nodes"
    )
    return {
        "community_count": len(communities),
        "algorithm": "louvain",
    }


def write_graph_store(nodes, edges, community_info):
    graph = {
        "metadata": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "source": "FAISS metadata chunks + Colab LLM extraction",
            "llm_model": MODEL_NAME,
            "max_chunks": MAX_CHUNKS or "all",
            "max_triples_per_chunk": MAX_TRIPLES_PER_CHUNK,
            "community_detection": community_info,
        },
        "nodes": nodes,
        "edges": edges,
    }
    Path("graph_store.json").write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return graph


def write_cypher(nodes, edges):
    lines = [
        "CREATE CONSTRAINT entity_name IF NOT EXISTS",
        "FOR (e:Entity) REQUIRE e.name IS UNIQUE;",
        "",
    ]
    for node in nodes:
        name = cypher_string(node["name"])
        label = node["label"]
        lines.append(
            f"MERGE (e:Entity {{name: '{name}'}}) "
            f"SET e:{label}, "
            f"e.mentions = {int(node['mentions'])}, "
            f"e.community = {int(node.get('community', -1))};"
        )

    lines.append("")
    for edge in edges:
        source = cypher_string(edge["source"])
        target = cypher_string(edge["target"])
        rel = relation_name(edge["relation"])
        weight = int(edge["weight"])
        chunk_ids = edge["chunk_ids"]
        lines.extend(
            [
                f"MERGE (s:Entity {{name: '{source}'}})",
                f"MERGE (o:Entity {{name: '{target}'}})",
                f"MERGE (s)-[r:{rel}]->(o)",
                f"SET r.weight = {weight}, r.chunk_ids = {chunk_ids};",
            ]
        )

    Path("neo4j_import.cypher").write_text("\n".join(lines), encoding="utf-8")


def split_cypher_script(script):
    statements = []
    current = []
    for line in script.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        current.append(line)
        if stripped.endswith(";"):
            statement = "\n".join(current).strip()
            statements.append(statement[:-1].strip())
            current = []
    if current:
        statements.append("\n".join(current).strip())
    return [statement for statement in statements if statement]


def import_to_neo4j(uri, username, password, database="neo4j"):
    from neo4j import GraphDatabase

    script = Path("neo4j_import.cypher").read_text(encoding="utf-8")
    statements = split_cypher_script(script)
    driver = GraphDatabase.driver(uri, auth=(username, password))

    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for idx, statement in enumerate(statements, start=1):
                session.run(statement).consume()
                if idx % 100 == 0:
                    print(f"Executed {idx}/{len(statements)} Cypher statements")
        print(f"Neo4j Aura import complete: {len(statements)} statements executed")
    finally:
        driver.close()


def main():
    index = faiss.read_index("vector_store.faiss")
    metadata = json.loads(Path("vector_store_metadata.json").read_text(encoding="utf-8"))
    chunks = metadata["chunks"]
    print(f"Loaded FAISS index: {index.ntotal} vectors, dim={index.d}")
    print(f"Loaded metadata chunks: {len(chunks)}")

    triples = extract_triples(chunks)
    nodes, edges = build_graph(triples)
    community_info = apply_louvain(nodes, edges)
    graph = write_graph_store(nodes, edges, community_info)
    write_cypher(nodes, edges)

    print("Graph store generated.")
    print(json.dumps(graph["metadata"], indent=2))
    print("Files: graph_store.json, neo4j_import.cypher")


if __name__ == "__main__":
    main()
