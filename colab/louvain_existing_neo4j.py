import json
from getpass import getpass
from collections import Counter
from pathlib import Path


def export_network_html(graph, partition, output_path="louvain_communities_network.html"):
    try:
        from pyvis.network import Network
        import math
        import networkx as nx
    except Exception as exc:
        print(f"PyVis unavailable, skipping HTML network export: {exc}")
        return

    net = Network(height="820px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.toggle_physics(False)

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
        "#005f73", "#ca6702", "#6a994e", "#9b2226", "#5a189a",
    ]

    communities = sorted(set(partition.values()))
    community_nodes = {
        community: [node for node, value in partition.items() if value == community]
        for community in communities
    }
    ranked = sorted(community_nodes, key=lambda c: len(community_nodes[c]), reverse=True)
    community_rank = {community: rank for rank, community in enumerate(ranked)}

    # Put each community around its own center, then lay nodes inside that island.
    centers = {}
    radius = 850
    for rank, community in enumerate(ranked):
        angle = 2 * math.pi * rank / max(len(ranked), 1)
        centers[community] = (radius * math.cos(angle), radius * math.sin(angle))

    positions = {}
    for community, nodes in community_nodes.items():
        subgraph = graph.subgraph(nodes)
        if len(nodes) == 1:
            local_positions = {nodes[0]: (0.0, 0.0)}
        else:
            local_positions = nx.spring_layout(
                subgraph,
                weight="weight",
                seed=42,
                k=1.2 / math.sqrt(max(len(nodes), 1)),
                iterations=100,
            )
        cx, cy = centers[community]
        scale = 220 + 10 * math.sqrt(len(nodes))
        for node_id, (x, y) in local_positions.items():
            positions[node_id] = (cx + x * scale, cy + y * scale)

    # Add large community anchor nodes to make the grouping obvious.
    for community in ranked:
        cx, cy = centers[community]
        size = len(community_nodes[community])
        color = palette[community_rank[community] % len(palette)]
        net.add_node(
            f"community-{community}",
            label=f"Community {community}\n{size} entities",
            title=f"Community {community}: {size} entities",
            x=cx,
            y=cy - 260,
            size=26,
            color=color,
            shape="box",
            physics=False,
            fixed=True,
        )

    for node_id, attrs in graph.nodes(data=True):
        community = int(partition.get(node_id, -1))
        rank = community_rank.get(community, 0)
        x, y = positions.get(node_id, (0, 0))
        label = attrs.get("name", node_id)
        net.add_node(
            node_id,
            label=label,
            title=f"{label}<br>Community: {community}",
            color=palette[rank % len(palette)] if community >= 0 else "#999999",
            group=community,
            x=x,
            y=y,
            size=8 + min(graph.degree(node_id), 18),
            physics=False,
            fixed=True,
        )

    for source, target, attrs in graph.edges(data=True):
        source_comm = partition.get(source, -1)
        target_comm = partition.get(target, -1)
        if source_comm == target_comm:
            net.add_edge(
                source,
                target,
                value=float(attrs.get("weight", 1)),
                color="#999999",
                width=0.8,
            )
        else:
            # Keep cross-community structure visible but faint.
            net.add_edge(
                source,
                target,
                value=0.2,
                color="rgba(160,160,160,0.18)",
                width=0.2,
                dashes=True,
            )

    net.write_html(output_path, notebook=False)
    add_community_controls(output_path, partition)
    print(f"Network visualization exported: {output_path}")


def add_community_controls(html_path, partition):
    path = Path(html_path)
    html = path.read_text(encoding="utf-8")
    counts = Counter(partition.values()).most_common()
    communities = [
        {"id": int(community), "size": int(size)}
        for community, size in counts
    ]
    controls = f"""
<style>
  #communityPanel {{
    position: fixed;
    top: 12px;
    left: 12px;
    width: 280px;
    max-height: 92vh;
    overflow: auto;
    z-index: 9999;
    background: rgba(255, 255, 255, 0.95);
    border: 1px solid #d0d7de;
    border-radius: 8px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.14);
    padding: 12px;
    font-family: Arial, sans-serif;
    color: #222;
  }}
  #communityPanel h3 {{
    margin: 0 0 8px;
    font-size: 15px;
  }}
  #communityPanel .hint {{
    font-size: 12px;
    color: #57606a;
    margin-bottom: 10px;
    line-height: 1.35;
  }}
  #communityPanel button {{
    display: inline-flex;
    align-items: center;
    justify-content: space-between;
    width: 100%;
    margin: 3px 0;
    padding: 7px 9px;
    border-radius: 6px;
    border: 1px solid #d0d7de;
    background: #f6f8fa;
    cursor: pointer;
    font-size: 12px;
  }}
  #communityPanel button:hover {{
    background: #eef2f6;
  }}
  #communityPanel button.active {{
    border-color: #0969da;
    background: #ddf4ff;
    font-weight: 700;
  }}
  #communityPanel .allBtn {{
    background: #24292f;
    color: white;
    border-color: #24292f;
  }}
  #communityPanel .meta {{
    color: #57606a;
  }}
</style>
<div id="communityPanel">
  <h3>Louvain Communities</h3>
  <div class="hint">Click a community to isolate it. Click "Show all" to restore the full graph.</div>
  <button class="allBtn active" onclick="showAllCommunities(this)">Show all <span class="meta">{len(partition)} nodes</span></button>
  <div id="communityButtons"></div>
</div>
<script>
  const COMMUNITY_DATA = {json.dumps(communities)};
  const NODE_COMMUNITY = {json.dumps({str(node): int(comm) for node, comm in partition.items()})};

  function getNetworkObjects() {{
    return {{ nodes: nodes, edges: edges, network: network }};
  }}

  function clearActiveButtons(activeButton) {{
    document.querySelectorAll('#communityPanel button').forEach(btn => btn.classList.remove('active'));
    if (activeButton) activeButton.classList.add('active');
  }}

  function showAllCommunities(button) {{
    const obj = getNetworkObjects();
    obj.nodes.update(obj.nodes.get().map(node => ({{
      id: node.id,
      hidden: false,
      label: node._originalLabel || node.label
    }})));
    obj.edges.update(obj.edges.get().map(edge => ({{
      id: edge.id,
      hidden: false
    }})));
    clearActiveButtons(button);
    obj.network.fit({{ animation: true }});
  }}

  function showCommunity(community, button) {{
    const obj = getNetworkObjects();
    const visible = new Set();
    obj.nodes.get().forEach(node => {{
      const comm = NODE_COMMUNITY[String(node.id)];
      const isAnchor = String(node.id) === `community-${{community}}`;
      if (comm === community || isAnchor) visible.add(node.id);
    }});

    obj.nodes.update(obj.nodes.get().map(node => ({{
      id: node.id,
      hidden: !visible.has(node.id),
      label: node._originalLabel || node.label
    }})));

    obj.edges.update(obj.edges.get().map(edge => ({{
      id: edge.id,
      hidden: !(visible.has(edge.from) && visible.has(edge.to))
    }})));

    clearActiveButtons(button);
    obj.network.fit({{ nodes: Array.from(visible), animation: true }});
  }}

  function buildCommunityButtons() {{
    const container = document.getElementById('communityButtons');
    COMMUNITY_DATA.forEach(item => {{
      const button = document.createElement('button');
      button.innerHTML = `<span>Community ${{item.id}}</span><span class="meta">${{item.size}} nodes</span>`;
      button.onclick = () => showCommunity(item.id, button);
      container.appendChild(button);
    }});
  }}

  window.addEventListener('load', () => {{
    setTimeout(() => {{
      const obj = getNetworkObjects();
      obj.nodes.update(obj.nodes.get().map(node => ({{
        id: node.id,
        _originalLabel: node.label
      }})));
      buildCommunityButtons();
    }}, 500);
  }});
</script>
"""
    html = html.replace("<body>", "<body>\n" + controls)
    path.write_text(html, encoding="utf-8")


def export_top_community_html(graph, partition, top_n=6):
    ranked = Counter(partition.values()).most_common(top_n)
    for community, _ in ranked:
        nodes = [node for node, value in partition.items() if value == community]
        subgraph = graph.subgraph(nodes).copy()
        export_network_html(
            subgraph,
            {node: community for node in subgraph.nodes},
            output_path=f"community_{community}_network.html",
        )


def export_community_summary_html(graph, partition, output_path="louvain_community_summary.html"):
    try:
        from pyvis.network import Network
    except Exception as exc:
        print(f"PyVis unavailable, skipping community summary export: {exc}")
        return

    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f",
    ]
    ranked = Counter(partition.values()).most_common()
    rank = {community: idx for idx, (community, _) in enumerate(ranked)}

    summary = Counter()
    for source, target in graph.edges():
        s_comm = partition.get(source)
        t_comm = partition.get(target)
        if s_comm is None or t_comm is None:
            continue
        if s_comm == t_comm:
            continue
        key = tuple(sorted((s_comm, t_comm)))
        summary[key] += 1

    net = Network(height="700px", width="100%", bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-12000, central_gravity=0.25, spring_length=180)

    for community, size in ranked:
        sample_names = [
            graph.nodes[node].get("name", node)
            for node, value in partition.items()
            if value == community
        ][:15]
        net.add_node(
            str(community),
            label=f"Community {community}\n{size} entities",
            title="<br>".join(sample_names),
            size=18 + min(size, 60),
            color=palette[rank[community] % len(palette)],
        )

    for (left, right), weight in summary.items():
        net.add_edge(str(left), str(right), value=weight, title=f"{weight} cross edges")

    net.write_html(output_path, notebook=False)
    print(f"Community summary exported: {output_path}")


def run_louvain_on_existing_neo4j(uri, username, password, database="neo4j"):
    from neo4j import GraphDatabase
    import networkx as nx
    try:
        from community import community_louvain
    except Exception:
        import community_louvain

    driver = GraphDatabase.driver(uri, auth=(username, password))
    graph = nx.Graph()

    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            node_rows = session.run(
                """
                MATCH (e:Entity)
                RETURN elementId(e) AS id, e.name AS name
                """
            )
            id_to_name = {}
            for row in node_rows:
                node_id = row["id"]
                name = row["name"] or node_id
                id_to_name[node_id] = name
                graph.add_node(node_id, name=name)

            edge_rows = session.run(
                """
                MATCH (a:Entity)-[r]->(b:Entity)
                RETURN elementId(a) AS source,
                       elementId(b) AS target,
                       coalesce(r.weight, 1) AS weight
                """
            )
            for row in edge_rows:
                graph.add_edge(
                    row["source"],
                    row["target"],
                    weight=max(float(row["weight"] or 1), 1.0),
                )

            print(
                f"Loaded existing Neo4j graph: "
                f"{graph.number_of_nodes()} nodes, {graph.number_of_edges()} relationships"
            )

            if graph.number_of_edges() == 0:
                print("No relationships found; Louvain needs edges.")
                return

            partition = community_louvain.best_partition(
                graph,
                weight="weight",
                random_state=42,
            )
            community_count = len(set(partition.values()))
            export_network_html(graph, partition)
            export_community_summary_html(graph, partition)
            export_top_community_html(graph, partition)

            updates = [
                {"id": node_id, "community": int(community)}
                for node_id, community in partition.items()
            ]

            session.run(
                """
                UNWIND $updates AS update
                MATCH (e:Entity)
                WHERE elementId(e) = update.id
                SET e.community = update.community
                """,
                updates=updates,
            ).consume()

            print(f"Louvain complete: {community_count} communities written to Neo4j")
    finally:
        driver.close()


if __name__ == "__main__":
    uri = input("Neo4j Aura URI, for example neo4j+s://xxxx.databases.neo4j.io: ")
    username = input("Neo4j username [neo4j]: ") or "neo4j"
    password = getpass("Neo4j password: ")
    database = input("Neo4j database [neo4j]: ") or "neo4j"
    run_louvain_on_existing_neo4j(uri, username, password, database)
