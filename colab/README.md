# Google Colab Graph RAG Pipeline

This folder matches the pipeline from the schema:

1. FastAPI creates the FAISS vector store locally.
2. FastAPI packages the FAISS index, metadata, and this Colab workflow.
3. Google Colab runs a free Hugging Face LLM for entity/relation extraction.
4. Colab runs Louvain community detection on the extracted graph.
5. Colab exports a graph store JSON and Neo4j Cypher import file.
6. Colab can connect to Neo4j Aura with the official Python driver and import the graph automatically.

## Local step

Run the FastAPI app, then call:

```text
POST /pipeline
POST /colab/package
```

Download `static/colab_graph_rag_package.zip`.

## Colab step

Open `agentic_graph_vector_rag_colab.ipynb` in Google Colab, upload the ZIP when asked, then run all cells.

Outputs:

```text
graph_store.json
neo4j_import.cypher
```

The notebook includes an automatic Aura import cell. Enter your Aura URI, username, and password when prompted.

## Louvain only, without recreating the graph

If your graph is already in Neo4j Aura and you only want to add communities, use:

```text
louvain_existing_neo4j.ipynb
```

It reads the existing graph, runs Louvain in Colab, and writes only the `community` property back to existing `Entity` nodes.
It also exports:

```text
louvain_community_summary.html
louvain_communities_network.html
community_0_network.html, community_1_network.html, ...
```

Open the summary file to see communities as distinct nodes, or open the communities file to see fixed community islands with nodes colored by detected Louvain community.
