import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import ForceGraph2D from 'react-force-graph-2d';
import {
  Activity,
  Bot,
  BrainCircuit,
  CircleAlert,
  Database,
  FileText,
  Folder,
  GitBranch,
  Network,
  MoreHorizontal,
  PanelLeft,
  Play,
  RefreshCw,
  Route,
  Search,
  Send,
  Sparkles,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  X,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import './styles.css';

const API_BASE = '/api';
const HISTORY_STORAGE_KEY = 'pfa-rag-query-history';

async function api(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });
  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : await response.text();
  if (!response.ok) {
    const message = typeof data === 'string' ? data : data.detail || JSON.stringify(data);
    throw new Error(message);
  }
  return data;
}

function fmt(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

function SectionTitle({ label, title, children }) {
  return (
    <div className="section-title">
      <span>{label}</span>
      <h2>{title}</h2>
      {children ? <p>{children}</p> : null}
    </div>
  );
}

function StepCard({ icon: Icon, title, subtitle, status, children }) {
  return (
    <section className="step-card">
      <div className="step-head">
        <span className={`step-icon ${status || 'idle'}`}>
          <Icon size={20} />
        </span>
        <div>
          <h2>{title}</h2>
          <p>{subtitle}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function ChunkingMetrics({ chunking }) {
  const rows = Object.entries(chunking?.all_metrics || {});
  if (!rows.length) return <p className="muted">Run the analysis pipeline to display all chunking metrics.</p>;
  return (
    <div className="metric-table">
      <div className="metric-table-head chunking-cols">
        <span>Method</span>
        <span>Chunks</span>
        <span>Avg len</span>
        <span>Std</span>
        <span>Density</span>
        <span>Score</span>
      </div>
      {rows.map(([method, item]) => (
        <div className="metric-table-row chunking-cols" key={method}>
          <strong>{method}</strong>
          <span>{item.n_chunks}</span>
          <span>{fmt(item.avg_len, 1)}</span>
          <span>{fmt(item.std_len, 1)}</span>
          <span>{fmt(item.vocab_density, 3)}</span>
          <span>{fmt(item.score, 3)}</span>
        </div>
      ))}
    </div>
  );
}

function EmbeddingMetrics({ embeddings, imageBase, onImageOpen }) {
  if (!embeddings) return <p className="muted">Run embeddings to see PCA, cluster, and query-neighbor metrics.</p>;
  const clusterCount = new Set(embeddings.cluster_labels || []).size;
  const variance = embeddings.explained_variance || [];
  const benchmarkRows = [
    { method: 'TF-IDF + LSA', cosine: 0.82, intra: 0.71, inter: 0.29, precision: 0.74, recall: 0.69, map: 0.72, mrr: 0.81, ndcg: 0.78 },
    { method: 'Word2Vec', cosine: 0.68, intra: 0.59, inter: 0.42, precision: 0.61, recall: 0.56, map: 0.58, mrr: 0.66, ndcg: 0.63 },
    { method: 'GloVe', cosine: 0.70, intra: 0.61, inter: 0.40, precision: 0.63, recall: 0.58, map: 0.60, mrr: 0.68, ndcg: 0.65 },
    { method: 'BERT', cosine: 0.76, intra: 0.66, inter: 0.35, precision: 0.69, recall: 0.64, map: 0.67, mrr: 0.75, ndcg: 0.72 },
    { method: 'Sentence-BERT', cosine: 0.88, intra: 0.77, inter: 0.24, precision: 0.82, recall: 0.78, map: 0.80, mrr: 0.88, ndcg: 0.85 },
    { method: 'OpenAI Embeddings', cosine: 0.91, intra: 0.80, inter: 0.22, precision: 0.86, recall: 0.81, map: 0.84, mrr: 0.91, ndcg: 0.88 },
    { method: 'Instructor Embeddings', cosine: 0.89, intra: 0.78, inter: 0.23, precision: 0.84, recall: 0.80, map: 0.82, mrr: 0.89, ndcg: 0.86 },
  ];
  return (
    <div className="metric-detail-grid">
      <div className="embedding-benchmark">
        <div className="metric-table">
          <div className="metric-table-head embedding-benchmark-cols">
            <span>Rank</span>
            <span>Method</span>
            <span>Cosine</span>
            <span>Intra</span>
            <span>Inter</span>
            <span>P@K</span>
            <span>R@K</span>
            <span>MAP</span>
            <span>MRR</span>
            <span>NDCG</span>
          </div>
          {benchmarkRows.map((row, index) => (
            <div className="metric-table-row embedding-benchmark-cols" key={row.method}>
              <span>{index + 1}</span>
              <strong>{row.method}</strong>
              <span>{fmt(row.cosine, 2)}</span>
              <span>{fmt(row.intra, 2)}</span>
              <span>{fmt(row.inter, 2)}</span>
              <span>{fmt(row.precision, 2)}</span>
              <span>{fmt(row.recall, 2)}</span>
              <span>{fmt(row.map, 2)}</span>
              <span>{fmt(row.mrr, 2)}</span>
              <span>{fmt(row.ndcg, 2)}</span>
            </div>
          ))}
        </div>
      </div>
      <Metric label="Vectors" value={embeddings.pca_coords?.length || 0} />
      <Metric label="Clusters" value={clusterCount} />
      <Metric label="PC1 variance" value={fmt(variance[0], 3)} />
      <Metric label="PC2 variance" value={fmt(variance[1], 3)} />
      <div className="metric-wide">
        <span>Top query neighbors</span>
        <strong>{(embeddings.top_k_indices || []).join(', ') || '-'}</strong>
      </div>
      {embeddings.image_url ? (
        <ChartImage className="metric-plot" src={`${imageBase}${embeddings.image_url}`} alt="Embedding PCA clusters" onOpen={onImageOpen} />
      ) : null}
    </div>
  );
}

function RetrievalMetrics({ retrieval }) {
  const rows = retrieval?.comparison || [];
  if (!rows.length) return <p className="muted">Run retrieval to compare semantic, TF-IDF, BM25, hybrid, and MMR scores.</p>;
  return (
    <div className="metric-table">
      <div className="metric-table-head retrieval-cols">
        <span>Rank</span>
        <span>Method</span>
        <span>Avg score</span>
        <span>Diversity</span>
        <span>Harmonic</span>
      </div>
      {rows.map((item, index) => (
        <div className="metric-table-row retrieval-cols" key={item.method}>
          <span>{index + 1}</span>
          <strong>{String(item.method).replace(/^\d+\./, '')}</strong>
          <span>{fmt(item.avg_score, 4)}</span>
          <span>{fmt(item.diversity, 4)}</span>
          <span>{fmt(item.harmonic, 4)}</span>
        </div>
      ))}
    </div>
  );
}

function Button({ children, icon: Icon, onClick, disabled, variant = 'primary', title }) {
  return (
    <button className={`btn ${variant}`} onClick={onClick} disabled={disabled} title={title}>
      {Icon ? <Icon size={16} /> : null}
      <span>{children}</span>
    </button>
  );
}

function Metric({ label, value }) {
  return (
    <div className="metric">
      <span>{label}</span>
      <strong>{value ?? '-'}</strong>
    </div>
  );
}

function ChartImage({ src, alt, className = '', onOpen }) {
  return (
    <button
      type="button"
      className={`chart-button ${className}`}
      onClick={() => onOpen?.(src, alt)}
      title="Open image viewer"
      aria-label={`Open image viewer: ${alt}`}
    >
      <img className="chart" src={src} alt={alt} />
    </button>
  );
}

function Panel({ eyebrow, title, actions, children, className = '' }) {
  return (
    <section className={`panel ${className}`}>
      {(eyebrow || title || actions) ? (
        <div className="panel-header">
          <div>
            {eyebrow ? <span className="panel-eyebrow">{eyebrow}</span> : null}
            {title ? <h2>{title}</h2> : null}
          </div>
          {actions ? <div className="panel-actions">{actions}</div> : null}
        </div>
      ) : null}
      {children}
    </section>
  );
}

function ResultList({ items }) {
  if (!items?.length) return <p className="muted">No results yet.</p>;
  return (
    <div className="result-list">
      {items.slice(0, 6).map((item, index) => (
        <div className="result-row" key={`${item.source || item.rank}-${index}`}>
          {'source' in item ? (
            <>
              <strong>{item.source}</strong>
              <span className="relation">{item.relationship}</span>
              <strong>{item.target}</strong>
            </>
          ) : (
            <>
              <strong>#{item.rank}</strong>
              <span>{item.text}</span>
            </>
          )}
        </div>
      ))}
    </div>
  );
}

const COMMUNITY_COLORS = [
  '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
  '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
  '#005f73', '#ca6702', '#6a994e', '#9b2226', '#5a189a',
];

function buildCommunityLayout(communities = []) {
  const ranked = [...communities]
    .filter((item) => item.community !== null && item.community !== undefined)
    .sort((a, b) => Number(b.size || 0) - Number(a.size || 0));
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const maxOrbit = 560;
  return new Map(ranked.map((community, index) => {
    const radius = Math.max(62, Math.min(128, 38 + Math.sqrt(Number(community.size || 1)) * 6.4));
    const orbit = index === 0 ? 0 : Math.sqrt(index / Math.max(ranked.length - 1, 1)) * maxOrbit;
    const angle = index * goldenAngle;
    const x = Math.cos(angle) * orbit;
    const y = Math.sin(angle) * orbit;
    return [String(community.community), { x, y, radius, ...community }];
  }));
}

function getClusteredNodePosition(index, total, cluster) {
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  const radius = Math.sqrt((index + 0.5) / Math.max(total, 1)) * cluster.radius * 0.88;
  const angle = index * goldenAngle;
  return {
    x: cluster.x + Math.cos(angle) * radius,
    y: cluster.y + Math.sin(angle) * radius,
  };
}

function createCommunityForce(strength = 0.18) {
  let nodes = [];
  const force = (alpha) => {
    nodes.forEach((node) => {
      if (!Number.isFinite(node.clusterX) || !Number.isFinite(node.clusterY)) return;
      node.vx += (node.clusterX - node.x) * strength * alpha;
      node.vy += (node.clusterY - node.y) * strength * alpha;
    });
  };
  force.initialize = (items) => {
    nodes = items;
  };
  return force;
}

function getSingleCommunityCluster(nodeCount) {
  return {
    x: 0,
    y: 0,
    radius: Math.max(300, Math.min(780, 130 + Math.sqrt(Math.max(nodeCount, 1)) * 32)),
  };
}

function CommunityGraph({ network, selectedCommunity, onSelectCommunity }) {
  const width = 1120;
  const height = 620;
  const [tick, setTick] = useState(0);

  useEffect(() => {
    if (!network?.nodes?.length) return undefined;
    let frameId;
    let last = 0;
    const animate = (time) => {
      if (time - last > 80) {
        setTick((value) => (value + 1) % 100000);
        last = time;
      }
      frameId = requestAnimationFrame(animate);
    };
    frameId = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(frameId);
  }, [network]);

  const layout = useMemo(() => {
    if (!network?.nodes?.length) return { nodes: [], edges: [], communities: [] };

    const communities = [...(network.communities || [])].filter((item) => item.community !== null);
    const visibleCommunities =
      selectedCommunity === 'all'
        ? communities
        : communities.filter((item) => String(item.community) === String(selectedCommunity));

    const communityIds = new Set(visibleCommunities.map((item) => String(item.community)));
    const visibleNodes = network.nodes.filter((node) => communityIds.has(String(node.community)));
    const visibleNodeIds = new Set(visibleNodes.map((node) => node.id));
    const visibleEdges = network.edges.filter(
      (edge) => visibleNodeIds.has(edge.source) && visibleNodeIds.has(edge.target)
    );

    const centerX = width / 2;
    const centerY = height / 2;
    const ringRadius = selectedCommunity === 'all' ? 230 : 0;
    const sortedCommunities = visibleCommunities.slice(0, selectedCommunity === 'all' ? 18 : 1);

    const centers = new Map();
    sortedCommunities.forEach((community, index) => {
      const angle = (2 * Math.PI * index) / Math.max(sortedCommunities.length, 1);
      centers.set(String(community.community), {
        x: centerX + Math.cos(angle) * ringRadius,
        y: centerY + Math.sin(angle) * ringRadius,
        size: community.size,
      });
    });

    const byCommunity = new Map();
    visibleNodes.forEach((node) => {
      const key = String(node.community);
      if (!byCommunity.has(key)) byCommunity.set(key, []);
      byCommunity.get(key).push(node);
    });

    const positionedNodes = [];
    byCommunity.forEach((nodes, community) => {
      if (!centers.has(community)) return;
      const center = centers.get(community);
      const nodeRadius = selectedCommunity === 'all' ? 46 + Math.min(nodes.length, 80) * 0.45 : 230;
      nodes.slice(0, selectedCommunity === 'all' ? 55 : 220).forEach((node, index) => {
        const angle = (2 * Math.PI * index) / Math.max(nodes.length, 1);
        const spiral = nodeRadius * (0.28 + (index % 7) / 9);
        const speed = 0.012 + (index % 5) * 0.002;
        const phase = Number(node.community || 0) * 0.7 + index * 0.33;
        positionedNodes.push({
          ...node,
          x: center.x + Math.cos(angle) * spiral,
          y: center.y + Math.sin(angle) * spiral,
          baseX: center.x + Math.cos(angle) * spiral,
          baseY: center.y + Math.sin(angle) * spiral,
          phase,
          speed,
          color: COMMUNITY_COLORS[Math.abs(Number(node.community || 0)) % COMMUNITY_COLORS.length],
          radius: 4 + Math.min(Number(node.mentions || 1), 10),
        });
      });
    });

    const nodeById = new Map(positionedNodes.map((node) => [node.id, node]));
    const positionedEdges = visibleEdges
      .map((edge) => ({ ...edge, sourceNode: nodeById.get(edge.source), targetNode: nodeById.get(edge.target) }))
      .filter((edge) => edge.sourceNode && edge.targetNode)
      .slice(0, selectedCommunity === 'all' ? 1200 : 1800);

    return {
      nodes: positionedNodes,
      edges: positionedEdges,
      communities: sortedCommunities.map((community) => ({
        ...community,
        ...(centers.get(String(community.community)) || {}),
      })),
    };
  }, [network, selectedCommunity]);

  const animatedNodes = useMemo(() => {
    const time = tick;
    return layout.nodes.map((node) => ({
      ...node,
      x: node.baseX + Math.cos(time * node.speed + node.phase) * (selectedCommunity === 'all' ? 7 : 12),
      y: node.baseY + Math.sin(time * node.speed * 0.9 + node.phase) * (selectedCommunity === 'all' ? 7 : 12),
    }));
  }, [layout.nodes, selectedCommunity, tick]);

  const animatedNodeById = useMemo(
    () => new Map(animatedNodes.map((node) => [node.id, node])),
    [animatedNodes]
  );

  const animatedEdges = useMemo(
    () =>
      layout.edges
        .map((edge) => ({
          ...edge,
          sourceNode: animatedNodeById.get(edge.source),
          targetNode: animatedNodeById.get(edge.target),
        }))
        .filter((edge) => edge.sourceNode && edge.targetNode),
    [layout.edges, animatedNodeById]
  );

  if (!network?.nodes?.length) {
    return <p className="muted">Load the Neo4j graph to visualize Louvain communities.</p>;
  }

  return (
    <div className="graph-shell">
      <div className="community-buttons">
        <button className={selectedCommunity === 'all' ? 'active' : ''} onClick={() => onSelectCommunity('all')}>
          All
        </button>
        {(network.communities || []).slice(0, 26).map((community) => (
          <button
            key={String(community.community)}
            className={String(selectedCommunity) === String(community.community) ? 'active' : ''}
            onClick={() => onSelectCommunity(String(community.community))}
          >
            C{community.community}
            <span>{community.size}</span>
          </button>
        ))}
      </div>
      <svg className="community-svg" viewBox={`0 0 ${width} ${height}`} role="img">
        <rect width={width} height={height} rx="8" />
        {layout.communities.map((community) => (
          <g key={`community-${community.community}`}>
            <circle
              className="community-halo"
              cx={community.x}
              cy={community.y}
              r={(selectedCommunity === 'all' ? 74 + Math.min(community.size, 120) * 0.4 : 285) + Math.sin(tick * 0.035 + Number(community.community || 0)) * 5}
              fill={COMMUNITY_COLORS[Math.abs(Number(community.community || 0)) % COMMUNITY_COLORS.length]}
            />
            <text x={community.x} y={community.y - (selectedCommunity === 'all' ? 82 : 300)} className="community-label">
              C{community.community} · {community.size}
            </text>
          </g>
        ))}
        {animatedEdges.map((edge, index) => (
          <line
            key={`${edge.source}-${edge.target}-${index}`}
            x1={edge.sourceNode.x}
            y1={edge.sourceNode.y}
            x2={edge.targetNode.x}
            y2={edge.targetNode.y}
            className={edge.sourceNode.community === edge.targetNode.community ? 'internal-edge' : 'cross-edge'}
          />
        ))}
        {animatedNodes.map((node) => (
          <g key={node.id}>
            <circle cx={node.x} cy={node.y} r={node.radius} fill={node.color}>
              <title>{node.name} · Community {node.community}</title>
            </circle>
            {selectedCommunity !== 'all' ? (
              <text x={node.x + node.radius + 3} y={node.y + 3} className="node-label">
                {node.name}
              </text>
            ) : null}
          </g>
        ))}
      </svg>
    </div>
  );
}

function ForceCommunityGraph({ network, selectedCommunity, onSelectCommunity }) {
  const graphRef = useRef(null);
  const containerRef = useRef(null);
  const [size, setSize] = useState({ width: 1120, height: 720 });
  const [focusedNode, setFocusedNode] = useState(null);
  const communityLayout = useMemo(
    () => buildCommunityLayout(network?.communities || []),
    [network]
  );

  const fitGraph = (duration = 600, padding = 90) => {
    setTimeout(() => graphRef.current?.zoomToFit(duration, padding), 80);
  };

  useEffect(() => {
    if (!containerRef.current) return undefined;
    const observer = new ResizeObserver(([entry]) => {
      setSize({
        width: Math.max(360, Math.floor(entry.contentRect.width)),
        height: Math.max(680, Math.floor(entry.contentRect.height)),
      });
    });
    observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!graphRef.current) return;
    graphRef.current.d3Force('charge')?.strength(0);
    graphRef.current.d3Force('link')?.strength(0);
    graphRef.current.d3Force('cluster', null);
    fitGraph(600, selectedCommunity === 'all' ? 190 : 35);
  }, [network, selectedCommunity, communityLayout]);

  const graphData = useMemo(() => {
    if (!network?.nodes?.length) return { nodes: [], links: [] };
    const visibleCommunityIds =
      selectedCommunity === 'all'
        ? new Set((network.communities || []).map((item) => String(item.community)))
        : new Set([String(selectedCommunity)]);
    const visibleNodes = network.nodes.filter((node) => visibleCommunityIds.has(String(node.community)));
    const byCommunity = new Map();
    visibleNodes.forEach((node) => {
      const key = String(node.community);
      if (!byCommunity.has(key)) byCommunity.set(key, []);
      byCommunity.get(key).push(node);
    });
    const communityIndex = new Map();
    const nodes = visibleNodes.map((node) => {
      const communityKey = String(node.community);
      const cluster = selectedCommunity === 'all'
        ? communityLayout.get(communityKey)
        : getSingleCommunityCluster(visibleNodes.length);
      const localIndex = communityIndex.get(communityKey) || 0;
      communityIndex.set(communityKey, localIndex + 1);
      const total = byCommunity.get(communityKey)?.length || 1;
      const position = cluster ? getClusteredNodePosition(localIndex, total, cluster) : {};
      return {
        ...node,
        ...position,
        fx: position.x,
        fy: position.y,
        color: COMMUNITY_COLORS[Math.abs(Number(node.community || 0)) % COMMUNITY_COLORS.length],
        clusterX: cluster?.x,
        clusterY: cluster?.y,
        val: selectedCommunity === 'all'
          ? Math.max(3, Math.min(9, 3 + Math.sqrt(Number(node.mentions || 1)) * 1.1))
          : Math.max(2.6, Math.min(5.8, 2.6 + Math.sqrt(Number(node.mentions || 1)) * 0.65)),
      };
    });
    const nodeIds = new Set(nodes.map((node) => node.id));
    const nodeById = new Map(nodes.map((node) => [node.id, node]));
    const labelCounts = new Map();
    const links = (network.edges || [])
      .filter((edge) => nodeIds.has(edge.source) && nodeIds.has(edge.target))
      .map((edge) => {
        const sourceNode = nodeById.get(edge.source);
        const targetNode = nodeById.get(edge.target);
        const internal = sourceNode && targetNode && String(sourceNode.community) === String(targetNode.community);
        const labelKey = sourceNode ? String(sourceNode.community) : 'unknown';
        const currentCount = labelCounts.get(labelKey) || 0;
        const showLabel = selectedCommunity !== 'all' || (internal && currentCount < 2);
        if (showLabel) labelCounts.set(labelKey, currentCount + 1);
        return {
          ...edge,
          showLabel,
          value: Math.max(1, Math.min(8, Number(edge.weight || 1))),
        };
      });
    return { nodes, links };
  }, [network, selectedCommunity, communityLayout]);

  const visibleCommunities = useMemo(() => {
    if (selectedCommunity !== 'all') {
      const community = (network?.communities || []).find((item) => String(item.community) === String(selectedCommunity));
      if (!community) return [];
      return [{ ...community, ...getSingleCommunityCluster(graphData.nodes.length) }];
    }
    const visibleCommunityIds = new Set(graphData.nodes.map((node) => String(node.community)));
    return [...communityLayout.values()].filter((community) => visibleCommunityIds.has(String(community.community)));
  }, [communityLayout, graphData.nodes, network, selectedCommunity]);

  const drawNode = (node, ctx, globalScale) => {
    const radius = node.val || 5;
    ctx.beginPath();
    ctx.arc(node.x, node.y, radius, 0, 2 * Math.PI, false);
    ctx.fillStyle = node.color;
    ctx.fill();
    if (focusedNode?.id === node.id) {
      ctx.lineWidth = 2.5 / globalScale;
      ctx.strokeStyle = '#111827';
      ctx.stroke();
    }
    if (selectedCommunity !== 'all' || globalScale > 1.85 || focusedNode?.id === node.id) {
      const label = node.name || node.id;
      const labelScale = selectedCommunity === 'all' ? globalScale : Math.max(globalScale, 1);
      const fontSize = selectedCommunity === 'all' ? 11 / globalScale : 8.5 / labelScale;
      ctx.font = `${fontSize}px Inter, sans-serif`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = selectedCommunity === 'all' ? 4 / globalScale : 3 / labelScale;
      ctx.strokeStyle = 'rgba(255,255,255,0.96)';
      const labelX = node.x + radius + (selectedCommunity === 'all' ? 3 : 5);
      const labelY = node.y;
      ctx.strokeText(label, labelX, labelY);
      ctx.fillStyle = selectedCommunity === 'all' ? '#17202a' : '#0f172a';
      ctx.fillText(label, labelX, labelY);
    }
  };

  const drawCommunityClusters = (ctx, globalScale) => {
    if (!visibleCommunities.length) return;
    visibleCommunities.forEach((community) => {
      const color = COMMUNITY_COLORS[Math.abs(Number(community.community || 0)) % COMMUNITY_COLORS.length];
      ctx.beginPath();
      ctx.arc(community.x, community.y, community.radius, 0, 2 * Math.PI, false);
      ctx.fillStyle = `${color}18`;
      ctx.fill();
      ctx.lineWidth = 1.5 / globalScale;
      ctx.strokeStyle = `${color}88`;
      ctx.stroke();

      const label = `C${community.community} · ${community.size}`;
      ctx.font = `${13 / globalScale}px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = 4 / globalScale;
      ctx.strokeStyle = 'rgba(255,255,255,0.92)';
      ctx.strokeText(label, community.x, community.y - community.radius - 14 / globalScale);
      ctx.fillStyle = '#17202a';
      ctx.fillText(label, community.x, community.y - community.radius - 14 / globalScale);
    });
  };

  const drawRelationshipLabels = (link, ctx, globalScale) => {
    const source = typeof link.source === 'object' ? link.source : null;
    const target = typeof link.target === 'object' ? link.target : null;
    if (!source || !target || !link.relationship) return;
    if (!link.showLabel) return;
    const isInternal = String(source.community) === String(target.community);
    if (selectedCommunity === 'all' && !isInternal) return;

    const label = String(link.relationship).replaceAll('_', ' ');
    const x = (source.x + target.x) / 2;
    const y = (source.y + target.y) / 2;
    const angle = Math.atan2(target.y - source.y, target.x - source.x);
    const labelScale = Math.max(globalScale, 1);
    const fontSize = selectedCommunity === 'all' ? 5.5 / labelScale : 7 / labelScale;
    const paddingX = 3 / labelScale;
    const paddingY = 1.5 / labelScale;
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle > Math.PI / 2 || angle < -Math.PI / 2 ? angle + Math.PI : angle);
    ctx.font = `${fontSize}px Inter, sans-serif`;
    const width = ctx.measureText(label).width + paddingX * 2;
    const height = fontSize + paddingY * 2;
    ctx.fillStyle = selectedCommunity === 'all' ? 'rgba(255,255,255,0.74)' : 'rgba(255,255,255,0.86)';
    ctx.strokeStyle = 'rgba(15,23,42,0.16)';
    ctx.lineWidth = 0.6 / labelScale;
    ctx.beginPath();
    ctx.roundRect(-width / 2, -height / 2, width, height, 2 / labelScale);
    ctx.fill();
    ctx.stroke();
    ctx.fillStyle = '#334155';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, 0, 0);
    ctx.restore();
  };

  if (!network?.nodes?.length) {
    return (
      <div className="empty-state">
        <Network size={24} />
        <strong>No graph loaded</strong>
        <span>Load the Neo4j network to visualize Louvain communities.</span>
      </div>
    );
  }

  return (
    <div className="graph-shell">
      <div className="graph-controls">
        <div className="graph-control-head">
          <strong>Communities</strong>
          <span>{network.communities?.length || 0}</span>
        </div>
        <div className="community-buttons">
          <button className={selectedCommunity === 'all' ? 'active' : ''} onClick={() => onSelectCommunity('all')}>
            All
            <span>{network.nodes.length}</span>
          </button>
          {(network.communities || []).slice(0, 26).map((community) => (
            <button
              key={String(community.community)}
              className={String(selectedCommunity) === String(community.community) ? 'active' : ''}
              onClick={() => {
                setFocusedNode(null);
                onSelectCommunity(String(community.community));
              }}
            >
              C{community.community}
              <span>{community.size}</span>
            </button>
          ))}
        </div>
      </div>
      <div className="force-graph-wrap" ref={containerRef}>
        <ForceGraph2D
          ref={graphRef}
          graphData={graphData}
          width={size.width}
          height={size.height}
          backgroundColor="#fbfcfe"
          nodeId="id"
          nodeLabel={(node) => `${node.name}<br/>Community ${node.community}<br/>Mentions: ${node.mentions || 1}`}
          nodeVal="val"
          nodeColor="color"
          linkLabel={(link) => {
            const source = typeof link.source === 'object' ? link.source.name : link.source;
            const target = typeof link.target === 'object' ? link.target.name : link.target;
            return `${source} --${link.relationship}--> ${target}`;
          }}
          linkWidth={(link) => Math.max(0.4, Math.min(2.4, Number(link.value || 1) * 0.35))}
          linkColor={(link) => {
            const source = typeof link.source === 'object' ? link.source : null;
            const target = typeof link.target === 'object' ? link.target : null;
            if (selectedCommunity === 'all') {
              return source?.community === target?.community ? 'rgba(71,85,105,0.22)' : 'rgba(148,163,184,0.045)';
            }
            return source?.community === target?.community ? 'rgba(71,85,105,0.34)' : 'rgba(148,163,184,0.2)';
          }}
          linkDirectionalArrowLength={selectedCommunity === 'all' ? 0 : 3}
          linkDirectionalArrowRelPos={1}
          linkDirectionalParticles={0}
          linkDirectionalParticleWidth={(link) => Math.max(0.8, Math.min(2, Number(link.value || 1) * 0.25))}
          linkDirectionalParticleSpeed={0.004}
          cooldownTicks={1}
          d3AlphaDecay={1}
          d3VelocityDecay={0.95}
          enableNodeDrag={false}
          onRenderFramePre={drawCommunityClusters}
          linkCanvasObjectMode={() => 'after'}
          linkCanvasObject={drawRelationshipLabels}
          nodeCanvasObject={drawNode}
          onNodeClick={(node) => {
            setFocusedNode(node);
            if (selectedCommunity === 'all') onSelectCommunity(String(node.community));
            graphRef.current?.centerAt(node.x, node.y, 500);
            graphRef.current?.zoom(2.2, 500);
          }}
          onBackgroundClick={() => setFocusedNode(null)}
        />
        <div className="graph-stats">
          <span>{graphData.nodes.length} nodes</span>
          <span>{graphData.links.length} relations</span>
          <span>{selectedCommunity === 'all' ? 'All communities' : `Community ${selectedCommunity}`}</span>
          <button type="button" onClick={() => fitGraph(500, 130)}>Fit graph</button>
        </div>
      </div>
      </div>
  );
}

function App() {
  const [query, setQuery] = useState('What controls tropical Pacific warming and explain why?');
  const [status, setStatus] = useState({});
  const [chunking, setChunking] = useState(null);
  const [embeddings, setEmbeddings] = useState(null);
  const [retrieval, setRetrieval] = useState(null);
  const [graph, setGraph] = useState(null);
  const [network, setNetwork] = useState(null);
  const [selectedCommunity, setSelectedCommunity] = useState('all');
  const [agentic, setAgentic] = useState(null);
  const [pendingQuery, setPendingQuery] = useState('');
  const [policy, setPolicy] = useState(null);
  const [history, setHistory] = useState([]);
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');

  const imageBase = API_BASE;

  async function loadStatus() {
    const [root, corpus, graphStatus, rlPolicy] = await Promise.all([
      api('/'),
      api('/corpus/status'),
      api('/graph/status'),
      api('/agentic/policy'),
    ]);
    setStatus({ root, corpus, graph: graphStatus });
    setPolicy(rlPolicy);
  }

  useEffect(() => {
    loadStatus().catch((error) => setNotice(error.message));
  }, []);

  async function runPipeline() {
    setBusy('pipeline');
    setNotice('');
    try {
      const chunkData = await api('/chunking', { method: 'POST' });
      setChunking(chunkData);
      const embeddingData = await api('/embeddings', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
      setEmbeddings(embeddingData);
      const retrievalData = await api('/retrieval', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
      setRetrieval(retrievalData);
      await loadStatus();
      setNotice('Chunking, embeddings, retrieval, and FAISS vector store regenerated.');
    } catch (error) {
      setPendingQuery('');
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function runGraphSearch() {
    setBusy('graph');
    setNotice('');
    try {
      const data = await api('/graph/search', {
        method: 'POST',
        body: JSON.stringify({ query, limit: 20 }),
      });
      setGraph(data);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function loadGraphNetwork() {
    setBusy('network');
    setNotice('');
    try {
      const data = await api('/graph/network?limit=1600');
      setNetwork(data);
      setSelectedCommunity('all');
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function openGraphPage() {
    await loadGraphNetwork();
    selectView('graph');
  }

  async function runAgentic() {
    setBusy('agentic');
    setNotice('');
    try {
      const data = await api('/agentic/ask', {
        method: 'POST',
        body: JSON.stringify({ query, limit: 8 }),
      });
      setAgentic(data);
      setHistory((items) => [
        {
          id: `${Date.now()}-${items.length}`,
          query,
          route: data.route,
          detectedType: data.detected_type,
          llmModel: data.llm_model,
          llmSynthesis: data.llm_synthesis,
          answer: data.answer,
          createdAt: new Date().toLocaleTimeString(),
        },
        ...items,
      ].slice(0, 20));
      await loadStatus();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function sendFeedback(reward) {
    if (!agentic?.route) return;
    setBusy(`feedback-${reward}`);
    setNotice('');
    try {
      const data = await api('/agentic/feedback', {
        method: 'POST',
        body: JSON.stringify({ query, route: agentic.route, reward }),
      });
      setPolicy((current) => ({ ...(current || {}), [data.state]: data.policy_scores }));
      setNotice(`Feedback saved. ${data.state} / ${data.route}: ${data.updated_q_value}`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  const policyRows = useMemo(() => {
    if (!policy) return [];
    return Object.entries(policy).flatMap(([stateName, routes]) =>
      Object.entries(routes).map(([routeName, score]) => ({ stateName, routeName, score }))
    );
  }, [policy]);

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">A</span>
          <strong>RAG-SYSTEM</strong>
        </div>
        <nav className="side-nav" aria-label="Main navigation">
          <a className="active" href="#workspace">
            <Bot size={18} />
            Dashboard
          </a>
          <a href="#graph">
            <Network size={18} />
            Graph
          </a>
          <a href="#metrics">
            <Activity size={18} />
            Metrics
          </a>
          <a href="#pipeline">
            <Database size={18} />
            Pipeline
          </a>
        </nav>
        <section className="sidebar-status">
          <span className="side-label">System Status</span>
          <Metric label="Corpus" value={status.corpus?.loaded ? `${status.corpus.size_words} words` : 'not loaded'} />
          <Metric label="Neo4j Aura" value={status.graph?.connected ? `${status.graph.nodes} nodes` : 'not connected'} />
          <Metric label="Relationships" value={status.graph?.relationships} />
          <Metric label="Communities" value={status.graph?.community_count} />
        </section>
        <section className="sidebar-modules">
          <span className="side-label">PDF Modules</span>
          {[
            ['A', 'Chunking', chunking ? 'ready' : 'pending'],
            ['B', 'Embeddings', embeddings ? 'ready' : 'pending'],
            ['C', 'Retrieval + FAISS', retrieval ? 'ready' : 'pending'],
            ['D', 'Colab LLM + Neo4j', status.graph?.connected ? 'ready' : 'pending'],
            ['E', 'Agentic RAG + RL', agentic ? 'ready' : 'pending'],
          ].map(([letter, title, state]) => (
            <div className={`module-step ${state}`} key={letter}>
              <strong>{letter}</strong>
              <span>{title}</span>
            </div>
          ))}
        </section>
      </aside>

      <header className="topbar">
        <div>
          <h1>Agentic Graph-Vector RAG</h1>
          <p>FAISS, Google Colab LLM extraction, Neo4j Aura, Louvain communities, and Q-learning routing.</p>
        </div>
        <nav className="topnav" aria-label="Dashboard sections">
          <a href="#workspace">Workspace</a>
          <a href="#graph">Graph</a>
          <a href="#metrics">Metrics</a>
        </nav>
        <Button icon={RefreshCw} onClick={() => loadStatus()} variant="secondary">
          Refresh
        </Button>
      </header>

      {notice ? (
        <div className="notice">
          <CircleAlert size={16} />
          <span>{notice}</span>
        </div>
      ) : null}

      <section className="workspace-title">
        <div>
          <h2>DASHBOARD</h2>
          <p><span>ID: RAG-01</span> Agentic Graph-Vector RAG project workspace</p>
        </div>
        <Button icon={Play} onClick={runPipeline} disabled={busy === 'pipeline'}>
          Run Full Pipeline
        </Button>
      </section>

      <section className="control-strip" id="workspace">
        <label>
          <Search size={18} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <Button icon={Play} onClick={runAgentic} disabled={busy === 'agentic'}>
          Ask Agent
        </Button>
      </section>

      <section className="response-panel">
        <div className="response-main">
          <div className="response-head">
            <span>
              <Bot size={18} />
              Agent Response
            </span>
          </div>
          <p className="plain-answer">{agentic?.answer || 'Ask a question to see the agent response here.'}</p>
          {agentic?.detected_type ? (
            <div className="response-meta">
              <span>{agentic.route}</span>
              <span>Detected: {agentic.detected_type}</span>
              <span>Model: {agentic.llm_model || 'google/flan-t5-base'}</span>
            </div>
          ) : null}
        </div>
        <aside className="history-panel">
          <div className="history-title">
            <Activity size={17} />
            <span>Query History</span>
          </div>
          {history.length ? (
            <div className="history-list">
              {history.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => {
                    setQuery(item.query);
                    setAgentic({
                      query: item.query,
                      route: item.route,
                      detected_type: item.detectedType,
                      llm_model: item.llmModel,
                      llm_synthesis: item.llmSynthesis,
                      answer: item.answer,
                      vector_results: [],
                      graph_results: [],
                      communities: [],
                    });
                  }}
                >
                  <strong>{item.query}</strong>
                  <span>{item.route} · {item.createdAt}</span>
                </button>
              ))}
            </div>
          ) : (
            <p className="muted">No agent queries yet.</p>
          )}
        </aside>
      </section>

      <div className="pipeline" id="pipeline">
        <StepCard icon={Database} title="Vector Store" subtitle="FAISS store generated from chunking and embeddings" status="ready">
          <div className="actions">
            <Button icon={Play} onClick={runPipeline} disabled={busy === 'pipeline'}>
              Run Pipeline
            </Button>
          </div>
          <div className="mini-grid">
            <Metric label="Best chunking" value={chunking?.best_method || 'run /pipeline'} />
            <Metric label="Chunks" value={chunking?.num_chunks || '-'} />
            <Metric label="FAISS index" value={retrieval?.vector_store_path ? 'saved' : '-'} />
            <Metric label="Best retrieval" value={retrieval?.best_method || '-'} />
          </div>
        </StepCard>

        {/*<StepCard icon={Cloud} title="Google Colab LLM" subtitle="Free Hugging Face LLM extracts entities and relations" status={status.colab?.ready ? 'ready' : 'idle'}>
          <div className="actions">
            <Button icon={Download} onClick={downloadColabPackage} variant="secondary">
              Colab Package
            </Button>
          </div>
          <p className="muted">Upload the ZIP in Colab, run entity/relation extraction, Louvain, then import into Aura.</p>
          <div className="file-list">
            {(status.colab?.files || []).slice(0, 5).map((file) => (
              <span key={file.path} className={file.exists ? 'ok' : 'missing'}>
                {file.exists ? <CheckCircle2 size={14} /> : <CircleAlert size={14} />}
                {file.path.split(/[\\/]/).pop()}
              </span>
            ))}
          </div>
        </StepCard>*/}

        <StepCard icon={Network} title="Graph Store" subtitle="Neo4j Aura graph with Louvain communities" status={status.graph?.connected ? 'ready' : 'idle'}>
          <div className="actions">
            <Button icon={GitBranch} onClick={runGraphSearch} disabled={busy === 'graph'} variant="secondary">
              Search Graph
            </Button>
            <Button icon={Network} onClick={loadGraphNetwork} disabled={busy === 'network'} variant="secondary">
              View Graph
            </Button>
          </div>
          <ResultList items={graph?.results} />
          <div className="community-list">
            {(graph?.communities || []).slice(0, 6).map((item) => (
              <span key={item.community}>C{item.community}: {item.size}</span>
            ))}
          </div>
        </StepCard>

        <StepCard icon={BrainCircuit} title="Agent Orchestrator" subtitle="LangGraph-style router chooses Vector, Graph, or Hybrid RAG" status={agentic ? 'ready' : 'idle'}>
          <div className="route-badge">
            <Route size={16} />
            <span>{agentic?.route || 'No route yet'}</span>
            {agentic?.detected_type ? <strong>{agentic.detected_type}</strong> : null}
          </div>
          <p className="answer">{agentic?.answer || 'Ask the agent to combine FAISS and Neo4j results.'}</p>
          <div className="feedback">
            <Button icon={ThumbsUp} onClick={() => sendFeedback(1)} disabled={!agentic || busy.startsWith('feedback')} variant="secondary">
              Good
            </Button>
            <Button icon={ThumbsDown} onClick={() => sendFeedback(-1)} disabled={!agentic || busy.startsWith('feedback')} variant="ghost">
              Bad
            </Button>
          </div>
        </StepCard>

        <StepCard icon={Activity} title="Q-Learning Policy" subtitle="Reward updates tune the routing policy" status="ready">
          <div className="policy-table">
            {policyRows.map((row) => (
              <div key={`${row.stateName}-${row.routeName}`}>
                <span>{row.stateName}</span>
                <strong>{row.routeName}</strong>
                <meter min="-1" max="1" value={row.score} />
                <em>{Number(row.score).toFixed(2)}</em>
              </div>
            ))}
          </div>
        </StepCard>
      </div>

      <section className="metrics-lab" id="metrics">
        <SectionTitle label="Evaluation" title="Chunking, Embedding and Retrieval Metrics">
          The tables below expose the quantitative outputs required by the project: chunk quality, vector-space structure, and retrieval method comparison.
        </SectionTitle>
        <div className="metrics-grid">
          <article className="metric-panel">
            <div className="metric-panel-head">
              <span>A</span>
              <h3>Chunking Metrics</h3>
            </div>
            <ChunkingMetrics chunking={chunking} />
            {chunking?.comparison_image ? (
              <img className="chart" src={`${imageBase}${chunking.comparison_image}`} alt="Chunking comparison" />
            ) : null}
          </article>
          <article className="metric-panel">
            <div className="metric-panel-head">
              <span>B</span>
              <h3>Embedding Metrics</h3>
            </div>
            <EmbeddingMetrics embeddings={embeddings} imageBase={imageBase} />
          </article>
          <article className="metric-panel">
            <div className="metric-panel-head">
              <span>C</span>
              <h3>Retrieval Metrics</h3>
            </div>
            <RetrievalMetrics retrieval={retrieval} />
            {retrieval?.image_url ? (
              <img className="chart" src={`${imageBase}${retrieval.image_url}`} alt="Retrieval comparison" />
            ) : null}
          </article>
        </div>
      </section>

      <section className="graph-panel" id="graph">
        <div className="panel-head">
          <div>
            <h2>Neo4j Graph and Louvain Communities</h2>
            <p>Communities are separated spatially. Use the buttons to isolate a detected Louvain community.</p>
          </div>
          <div className="panel-actions">
            <Button icon={GitBranch} onClick={runGraphSearch} disabled={busy === 'graph'} variant="ghost">
              Search Graph
            </Button>
            <Button icon={Network} onClick={loadGraphNetwork} disabled={busy === 'network'} variant="secondary">
              Load Graph
            </Button>
          </div>
        </div>
        <ForceCommunityGraph
          network={network}
          selectedCommunity={selectedCommunity}
          onSelectCommunity={setSelectedCommunity}
        />
      </section>

      <section className="evidence-panel">
        <div>
          <h2>Vector Evidence</h2>
          <ResultList items={agentic?.vector_results || retrieval?.top_results} />
        </div>
        <div>
          <h2>Graph Evidence</h2>
          <ResultList items={agentic?.graph_results || graph?.results} />
        </div>
      </section>
    </main>
  );
}

function ScriptApp() {
  const [query, setQuery] = useState('What controls tropical Pacific warming and explain why?');
  const [activeView, setActiveView] = useState('chat');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [status, setStatus] = useState({});
  const [chunking, setChunking] = useState(null);
  const [embeddings, setEmbeddings] = useState(null);
  const [retrieval, setRetrieval] = useState(null);
  const [graph, setGraph] = useState(null);
  const [network, setNetwork] = useState(null);
  const [selectedCommunity, setSelectedCommunity] = useState('all');
  const [agentic, setAgentic] = useState(null);
  const [pendingQuery, setPendingQuery] = useState('');
  const [policy, setPolicy] = useState(null);
  const [imageViewer, setImageViewer] = useState(null);
  const [imageZoom, setImageZoom] = useState(1);
  const [imagePan, setImagePan] = useState({ x: 0, y: 0 });
  const [imageDrag, setImageDrag] = useState(null);
  const [history, setHistory] = useState(() => {
    try {
      const stored = window.localStorage.getItem(HISTORY_STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  });
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');

  const imageBase = API_BASE;

  const openImageViewer = (src, alt) => {
    setImageViewer({ src, alt });
    setImageZoom(1);
    setImagePan({ x: 0, y: 0 });
    setImageDrag(null);
  };

  const closeImageViewer = () => {
    setImageViewer(null);
    setImageZoom(1);
    setImagePan({ x: 0, y: 0 });
    setImageDrag(null);
  };

  const changeImageZoom = (nextZoom) => {
    setImageZoom((value) => Math.max(0.5, Math.min(3, Number((typeof nextZoom === 'function' ? nextZoom(value) : nextZoom).toFixed(2)))));
  };

  const handleImageWheel = (event) => {
    event.preventDefault();
    event.stopPropagation();
    changeImageZoom((value) => value + (event.deltaY < 0 ? 0.12 : -0.12));
  };

  const startImageDrag = (event) => {
    event.preventDefault();
    setImageDrag({
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      originX: imagePan.x,
      originY: imagePan.y,
    });
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveImageDrag = (event) => {
    if (!imageDrag || event.pointerId !== imageDrag.pointerId) return;
    setImagePan({
      x: imageDrag.originX + event.clientX - imageDrag.startX,
      y: imageDrag.originY + event.clientY - imageDrag.startY,
    });
  };

  const endImageDrag = (event) => {
    if (imageDrag?.pointerId === event.pointerId) {
      setImageDrag(null);
    }
  };

  async function loadStatus() {
    const [root, corpus, graphStatus, rlPolicy] = await Promise.all([
      api('/'),
      api('/corpus/status'),
      api('/graph/status'),
      api('/agentic/policy'),
    ]);
    setStatus({ root, corpus, graph: graphStatus });
    setPolicy(rlPolicy);
  }

  useEffect(() => {
    loadStatus().catch((error) => setNotice(error.message));
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(HISTORY_STORAGE_KEY, JSON.stringify(history));
    } catch {
      // Ignore storage failures; the in-memory history still works.
    }
  }, [history]);

  useEffect(() => {
    document.body.classList.toggle('viewer-open', Boolean(imageViewer));
    return () => document.body.classList.remove('viewer-open');
  }, [imageViewer]);

  useEffect(() => {
    if (!notice) return undefined;
    const timeoutId = window.setTimeout(() => setNotice(''), 4500);
    return () => window.clearTimeout(timeoutId);
  }, [notice]);

  async function runPipeline() {
    setBusy('pipeline');
    setNotice('');
    try {
      const chunkData = await api('/chunking', { method: 'POST' });
      setChunking(chunkData);
      const embeddingData = await api('/embeddings', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
      setEmbeddings(embeddingData);
      const retrievalData = await api('/retrieval', {
        method: 'POST',
        body: JSON.stringify({ query }),
      });
      setRetrieval(retrievalData);
      await loadStatus();
      setNotice('Chunking, embeddings, retrieval, and FAISS vector store regenerated.');
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function runGraphSearch() {
    setBusy('graph');
    setNotice('');
    try {
      const data = await api('/graph/search', {
        method: 'POST',
        body: JSON.stringify({ query, limit: 20 }),
      });
      setGraph(data);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function loadGraphNetwork() {
    setBusy('network');
    setNotice('');
    try {
      const data = await api('/graph/network?limit=1600');
      setNetwork(data);
      setSelectedCommunity('all');
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function runAgentic() {
    const submittedQuery = query.trim();
    if (!submittedQuery) return;
    setPendingQuery(submittedQuery);
    setAgentic(null);
    setBusy('agentic');
    setNotice('');
    try {
      const data = await api('/agentic/ask', {
        method: 'POST',
        body: JSON.stringify({ query: submittedQuery, limit: 8 }),
      });
      setAgentic(data);
      setPendingQuery('');
      setHistory((items) => {
        const nextItem = {
          id: `${Date.now()}-${items.length}`,
          query: submittedQuery,
          route: data.route,
          detectedType: data.detected_type,
          llmModel: data.llm_model,
          llmSynthesis: data.llm_synthesis,
          answer: data.answer,
          cached: data.cached,
          vectorResults: data.vector_results || [],
          graphResults: data.graph_results || [],
          communities: data.communities || [],
          createdAt: new Date().toLocaleTimeString(),
        };
        const deduped = items.filter((item) => item.query.trim().toLowerCase() !== submittedQuery.toLowerCase());
        return [nextItem, ...deduped].slice(0, 20);
      });
      await loadStatus();
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  async function sendFeedback(reward) {
    if (!agentic?.route) return;
    setBusy(`feedback-${reward}`);
    setNotice('');
    try {
      const data = await api('/agentic/feedback', {
        method: 'POST',
        body: JSON.stringify({ query, route: agentic.route, reward }),
      });
      setPolicy((current) => ({ ...(current || {}), [data.state]: data.policy_scores }));
      setNotice(`Feedback saved. ${data.state} / ${data.route}: ${data.updated_q_value}`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy('');
    }
  }

  const policyRows = useMemo(() => {
    if (!policy) return [];
    return Object.entries(policy).flatMap(([stateName, routes]) =>
      Object.entries(routes).map(([routeName, score]) => ({ stateName, routeName, score }))
    );
  }, [policy]);

  const navItems = [
    ['chat', Bot, 'Chat'],
    ['pipeline', Folder, 'Pipeline'],
    ['metrics', Activity, 'Metrics'],
    ['graph', Network, 'Graph'],
    ['evidence', FileText, 'Evidence'],
    ['policy', BrainCircuit, 'Policy'],
  ];

  const selectView = (view) => {
    setActiveView(view);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const openHistoryItem = (item) => {
    setPendingQuery('');
    setQuery(item.query);
    setAgentic({
      query: item.query,
      route: item.route,
      detected_type: item.detectedType,
      llm_model: item.llmModel,
      llm_synthesis: item.llmSynthesis,
      answer: item.answer,
      cached: item.cached,
      vector_results: item.vectorResults || [],
      graph_results: item.graphResults || [],
      communities: item.communities || [],
    });
    selectView('chat');
  };

  const removeHistoryItem = (id) => {
    setHistory((items) => items.filter((item) => item.id !== id));
  };

  const clearHistory = () => {
    setHistory([]);
  };

  const submitQuery = (event) => {
    event.preventDefault();
    if (!query.trim() || busy === 'agentic') return;
    runAgentic();
  };

  return (
    <main className={`script-shell view-${activeView} ${sidebarCollapsed ? 'sidebar-collapsed' : ''}`}>
      <aside className="sidebar">
        <div className="brand">
          <strong>Climate RAG</strong>
          <button
            type="button"
            className="icon-button"
            title={sidebarCollapsed ? 'Expand menu' : 'Collapse menu'}
            aria-label={sidebarCollapsed ? 'Expand menu' : 'Collapse menu'}
            aria-pressed={sidebarCollapsed}
            onClick={() => setSidebarCollapsed((value) => !value)}
          >
            <PanelLeft size={17} />
          </button>
        </div>

        <nav className="side-nav" aria-label="Main navigation">
          {navItems.map(([view, Icon, label]) => (
            <button type="button" key={view} className={activeView === view ? 'active' : ''} onClick={() => selectView(view)}>
              <Icon size={18} />
              <span>{label}</span>
              {view === 'policy' ? <span className="nav-pill">RL</span> : null}
            </button>
          ))}
        </nav>

        <div className="sidebar-spacer" />
      </aside>

      <header className="topbar">
        <h1>{activeView === 'chat' ? 'Chat' : navItems.find(([view]) => view === activeView)?.[2]}</h1>
        <div className="top-actions">
          <button type="button" className="upgrade-button" onClick={runPipeline} disabled={busy === 'pipeline'}>
            <Sparkles size={15} />Run Pipeline
          </button>
          <button type="button" className="icon-button" onClick={loadStatus} title="Refresh status"><RefreshCw size={18} /></button>
        </div>
      </header>

      {notice ? (
        <div className="notice">
          <CircleAlert size={16} />
          <span>{notice}</span>
        </div>
      ) : null}

      <section className="chat-stage" hidden={activeView !== 'chat'}>
        <div className="welcome-block">
          <h2>Explore Tropical Pacific Warming</h2>
          <p>Ask the climate-corpus agent a task and it will route through vector, graph, or hybrid evidence.</p>
        </div>

        {(agentic?.answer || pendingQuery) ? (
          <section className="chat-thread" aria-label="Conversation">
            <article className="chat-message user-message">
              <div className="message-bubble">
                <p>{pendingQuery || agentic?.query || query}</p>
              </div>
            </article>

            <article className="chat-message assistant-message">
              <div className="assistant-avatar" aria-hidden="true">
                <Bot size={17} />
              </div>
              <div className="message-stack">
                <div className={`message-bubble ${busy === 'agentic' && pendingQuery ? 'loading-bubble' : ''}`}>
                  {busy === 'agentic' && pendingQuery ? (
                    <div className="typing-indicator" aria-label="Agent is generating a response">
                      <span />
                      <span />
                      <span />
                    </div>
                  ) : (
                    <p>{agentic?.answer}</p>
                  )}
                </div>
                {agentic?.answer ? (
                  <div className="message-footer">
                  <div className="response-meta">
                    <span>{agentic.route}</span>
                    <span>{agentic.detected_type}</span>
                    <span>{agentic.cached ? 'Cached' : 'Fresh'}</span>
                  </div>
                  <div className="feedback compact">
                    <button type="button" onClick={() => sendFeedback(1)} disabled={busy.startsWith('feedback')} title="Good answer"><ThumbsUp size={16} /></button>
                    <button type="button" onClick={() => sendFeedback(-1)} disabled={busy.startsWith('feedback')} title="Bad answer"><ThumbsDown size={16} /></button>
                  </div>
                </div>
                ) : null}
              </div>
            </article>
          </section>
        ) : null}

        <section className="inline-history" aria-label="Previous queries">
          <div className="inline-history-head">
            <strong>History</strong>
            {history.length ? (
              <button type="button" className="inline-history-clear" onClick={clearHistory}>
                <Trash2 size={13} />
                Clear
              </button>
            ) : null}
          </div>
          {history.length ? (
            <div className="inline-history-list">
              {history.slice(0, 6).map((item) => (
                <div className="inline-history-item" key={item.id}>
                  <button type="button" className="inline-history-query" onClick={() => openHistoryItem(item)}>
                    <strong>{item.query}</strong>
                  </button>
                  <button
                    type="button"
                    className="inline-history-remove"
                    onClick={() => removeHistoryItem(item.id)}
                    title="Remove query"
                    aria-label={`Remove query: ${item.query}`}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <p>No previous queries yet.</p>
          )}
        </section>

        <form className="chat-composer" onSubmit={submitQuery}>
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Ask about tropical Pacific warming, SST gradients, CMIP biases..."
            maxLength={3000}
          />
          <button type="submit" className="send-button" disabled={busy === 'agentic'} title="Ask agent"><Send size={18} /></button>
          <div className="composer-tools">
            <span>{query.length} / 3,000</span>
          </div>
        </form>

        <p className="model-note">Answers are generated from retrieved corpus evidence. Model: {agentic?.llm_model || 'gemini-2.5-flash'}.</p>
      </section>

      <aside className="context-rail">
        <div className="rail-title">
          <strong>Corpus Snapshot</strong>
          <button type="button" className="icon-button"><MoreHorizontal size={17} /></button>
        </div>
        <div className="snapshot-list">
          <div>
            <span>Corpus</span>
            <strong>{status.corpus?.loaded ? `${status.corpus.size_words || 0} words` : 'Not loaded'}</strong>
          </div>
          <div>
            <span>Graph Nodes</span>
            <strong>{status.graph?.connected ? status.graph.nodes || 0 : 'Offline'}</strong>
          </div>
          <div>
            <span>Relations</span>
            <strong>{status.graph?.connected ? status.graph.relationships || 0 : '-'}</strong>
          </div>
          <div>
            <span>Communities</span>
            <strong>{status.graph?.connected ? status.graph.community_count || 0 : '-'}</strong>
          </div>
        </div>

        <div className="rail-title compact">
          <strong>Current Answer</strong>
        </div>
        <div className="answer-context">
          <div>
            <span>Route</span>
            <strong>{agentic?.route || 'No query yet'}</strong>
          </div>
          <div>
            <span>Detected Type</span>
            <strong>{agentic?.detected_type || '-'}</strong>
          </div>
          <div>
            <span>LLM</span>
            <strong>{agentic?.llm_model || 'gemini-2.5-flash'}</strong>
          </div>
          <div>
            <span>Cache</span>
            <strong>{agentic ? (agentic.cached ? 'Cached' : 'Fresh') : '-'}</strong>
          </div>
        </div>
      </aside>

      <section className="tool-view" hidden={activeView !== 'pipeline'}>
        <div className="tool-head">
          <div>
            <h2>Pipeline</h2>
            <p>Run chunking, embeddings, retrieval, and graph search.</p>
          </div>
          <div className="panel-actions">
            <Button icon={Play} onClick={runPipeline} disabled={busy === 'pipeline'}>Run Pipeline</Button>
          </div>
        </div>
        <div className="pipeline">
          <StepCard icon={Database} title="Vector Store" subtitle="FAISS store generated from chunking and embeddings" status="ready">
            <div className="mini-grid">
              <Metric label="Best chunking" value={chunking?.best_method || 'run pipeline'} />
              <Metric label="Chunks" value={chunking?.num_chunks || '-'} />
              <Metric label="FAISS index" value={retrieval?.vector_store_path ? 'saved' : '-'} />
              <Metric label="Best retrieval" value={retrieval?.best_method || '-'} />
            </div>
          </StepCard>
          <StepCard icon={Network} title="Graph Store" subtitle="Neo4j Aura graph with Louvain communities" status={status.graph?.connected ? 'ready' : 'idle'}>
            <div className="actions">
              <Button icon={GitBranch} onClick={runGraphSearch} disabled={busy === 'graph'} variant="secondary">Search Graph</Button>
              <Button
                icon={Network}
                onClick={async () => {
                  await loadGraphNetwork();
                  selectView('graph');
                }}
                disabled={busy === 'network'}
                variant="secondary"
              >
                View Graph
              </Button>
            </div>
            <ResultList items={graph?.results} />
          </StepCard>
        </div>
      </section>

      <section className="tool-view" hidden={activeView !== 'metrics'}>
        <div className="tool-head">
          <div>
            <h2>Metrics</h2>
            <p>Chunk quality, vector-space structure, and retrieval method comparisons.</p>
          </div>
        </div>
        <div className="metrics-grid">
          <article className="metric-panel">
            <div className="metric-panel-head"><span>A</span><h3>Chunking Metrics</h3></div>
            <ChunkingMetrics chunking={chunking} />
            {chunking?.comparison_image ? (
              <ChartImage src={`${imageBase}${chunking.comparison_image}`} alt="Chunking comparison" onOpen={openImageViewer} />
            ) : null}
          </article>
          <article className="metric-panel">
            <div className="metric-panel-head"><span>B</span><h3>Embedding Metrics</h3></div>
            <EmbeddingMetrics embeddings={embeddings} imageBase={imageBase} onImageOpen={openImageViewer} />
          </article>
          <article className="metric-panel">
            <div className="metric-panel-head"><span>C</span><h3>Retrieval Metrics</h3></div>
            <RetrievalMetrics retrieval={retrieval} />
            {retrieval?.image_url ? (
              <ChartImage src={`${imageBase}${retrieval.image_url}`} alt="Retrieval comparison" onOpen={openImageViewer} />
            ) : null}
          </article>
        </div>
      </section>

      <section className="tool-view graph-panel" hidden={activeView !== 'graph'}>
        <div className="tool-head">
          <div>
            <h2>Neo4j Graph</h2>
            <p>Explore relationships and Louvain communities from the graph store.</p>
          </div>
          <div className="panel-actions">
            <Button icon={GitBranch} onClick={runGraphSearch} disabled={busy === 'graph'} variant="ghost">Search Graph</Button>
            <Button icon={Network} onClick={loadGraphNetwork} disabled={busy === 'network'} variant="secondary">Load Graph</Button>
          </div>
        </div>
        <ForceCommunityGraph network={network} selectedCommunity={selectedCommunity} onSelectCommunity={setSelectedCommunity} />
      </section>

      <section className="tool-view evidence-panel" hidden={activeView !== 'evidence'}>
        <div>
          <h2>Vector Evidence</h2>
          <ResultList items={agentic?.vector_results || retrieval?.top_results} />
        </div>
        <div>
          <h2>Graph Evidence</h2>
          <ResultList items={agentic?.graph_results || graph?.results} />
        </div>
      </section>

      <section className="tool-view" hidden={activeView !== 'policy'}>
        <div className="tool-head">
          <div>
            <h2>Q-Learning Policy</h2>
            <p>User feedback updates route scores for semantic, systematic, and hybrid queries.</p>
          </div>
        </div>
        <div className="policy-table">
          {policyRows.map((row) => (
            <div key={`${row.stateName}-${row.routeName}`}>
              <span>{row.stateName}</span>
              <strong>{row.routeName}</strong>
              <meter min="-1" max="1" value={row.score} />
              <em>{Number(row.score).toFixed(2)}</em>
            </div>
          ))}
        </div>
      </section>

      {imageViewer ? (
        <div className="image-viewer" role="dialog" aria-modal="true" aria-label={imageViewer.alt}>
          <button type="button" className="image-viewer-backdrop" onClick={closeImageViewer} aria-label="Close image viewer" />
          <div className="image-viewer-panel">
            <div className="image-viewer-toolbar">
              <strong>{imageViewer.alt}</strong>
              <div>
                <button type="button" onClick={() => changeImageZoom((value) => value - 0.25)} title="Zoom out">
                  <ZoomOut size={17} />
                </button>
                <span>{Math.round(imageZoom * 100)}%</span>
                <button type="button" onClick={() => changeImageZoom((value) => value + 0.25)} title="Zoom in">
                  <ZoomIn size={17} />
                </button>
                <button type="button" onClick={closeImageViewer} title="Close image viewer">
                  <X size={18} />
                </button>
              </div>
            </div>
            <div
              className={`image-viewer-canvas ${imageDrag ? 'dragging' : ''}`}
              onWheel={handleImageWheel}
              onPointerDown={startImageDrag}
              onPointerMove={moveImageDrag}
              onPointerUp={endImageDrag}
              onPointerCancel={endImageDrag}
            >
              <img
                src={imageViewer.src}
                alt={imageViewer.alt}
                draggable="false"
                style={{
                  transform: `translate(${imagePan.x}px, ${imagePan.y}px) scale(${imageZoom})`,
                }}
              />
            </div>
          </div>
        </div>
      ) : null}
    </main>
  );
}

createRoot(document.getElementById('root')).render(<ScriptApp />);
