import React, { useEffect, useMemo, useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

type JobState = "queued" | "running" | "completed" | "failed";
type NodeKind = "input" | "attention" | "mlp" | "output";

type NodeStatus = {
  kind: NodeKind;
  ready: boolean;
  rank: number | null;
};

type JobStatus = {
  job_id: string;
  state: JobState;
  prompt: string;
  expected_answer: string;
  output: string | null;
  is_correct: boolean | null;
  n_layers: number | null;
  n_heads: number | null;
  completed_nodes: number;
  total_nodes: number | null;
  current_step: string;
  error: string | null;
  nodes: Record<string, NodeStatus>;
};

type NodeDetail = {
  node: string;
  kind: NodeKind;
  rank: number | null;
  attention_image: string | null;
  logits_image: string | null;
};

type PromptSample = {
  prompt: string;
  subject: string;
  expected_answer: string;
  keywords: string;
};

type GraphNode = {
  name: string;
  node?: NodeStatus;
  x: number;
  y: number;
  width: number;
  height: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

function rankColor(rank: number | null, vocabThreshold = 5025) {
  if (rank === null || rank >= vocabThreshold) return "#ffffff";
  const normalized = Math.log(rank + 1e-8) / Math.log(vocabThreshold + 1e-8);
  const channel = Math.max(0, Math.min(255, Math.round(255 * normalized)));
  return `rgb(${channel}, 255, ${channel})`;
}

function App() {
  const [prompt, setPrompt] = useState("Sendai is located in the country of");
  const [expectedAnswer, setExpectedAnswer] = useState("Japan");
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<JobStatus | null>(null);
  const [selected, setSelected] = useState<NodeDetail | null>(null);
  const [samples, setSamples] = useState<PromptSample[]>([]);
  const [isSamplesOpen, setIsSamplesOpen] = useState(false);
  const [isLoadingSamples, setIsLoadingSamples] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;

    async function poll() {
      const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
      if (!response.ok) return;
      const nextStatus = (await response.json()) as JobStatus;
      if (!cancelled) setStatus(nextStatus);
    }

    poll();
    const intervalId = window.setInterval(poll, 1200);
    return () => {
      cancelled = true;
      window.clearInterval(intervalId);
    };
  }, [jobId]);

  const progress = useMemo(() => {
    if (!status?.total_nodes) return 0;
    return Math.round((status.completed_nodes / status.total_nodes) * 100);
  }, [status]);

  async function startAnalysis(event: React.FormEvent) {
    event.preventDefault();
    setMessage(null);
    setSelected(null);
    setStatus(null);
    setIsStarting(true);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, expected_answer: expectedAnswer }),
      });
      if (!response.ok) throw new Error("分析を開始できませんでした");
      const data = (await response.json()) as { job_id: string };
      setJobId(data.job_id);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "分析を開始できませんでした");
    } finally {
      setIsStarting(false);
    }
  }

  async function fillRandomPrompt() {
    setMessage(null);
    const response = await fetch(`${API_BASE}/api/random-prompt`);
    if (!response.ok) {
      setMessage("サンプルを取得できませんでした");
      return;
    }
    const data = (await response.json()) as { prompt: string; expected_answer: string };
    setPrompt(data.prompt);
    setExpectedAnswer(data.expected_answer);
  }

  async function openSamples() {
    setMessage(null);
    setIsSamplesOpen(true);
    if (samples.length > 0) return;

    setIsLoadingSamples(true);
    try {
      const response = await fetch(`${API_BASE}/api/prompt-samples`);
      if (!response.ok) throw new Error("サンプル一覧を取得できませんでした");
      setSamples((await response.json()) as PromptSample[]);
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "サンプル一覧を取得できませんでした",
      );
      setIsSamplesOpen(false);
    } finally {
      setIsLoadingSamples(false);
    }
  }

  function selectSample(sample: PromptSample) {
    setPrompt(sample.prompt);
    setExpectedAnswer(sample.expected_answer);
    setIsSamplesOpen(false);
  }

  async function openNode(nodeName: string, node: NodeStatus) {
    if (!jobId || !node.ready) return;
    setMessage(null);
    const response = await fetch(
      `${API_BASE}/api/jobs/${jobId}/nodes/${encodeURIComponent(nodeName)}`,
    );
    if (response.status === 202) {
      setMessage(`${nodeName} はまだ生成中です`);
      return;
    }
    if (!response.ok) {
      setMessage(`${nodeName} を開けませんでした`);
      return;
    }
    setSelected((await response.json()) as NodeDetail);
  }

  return (
    <main>
      <section className="intro">
        <div>
          <p className="eyebrow">Visualize LLM Demo</p>
          <h1>次の単語を予測する仕組み</h1>
        </div>
        <form className="controls" onSubmit={startAnalysis}>
          <label>
            入力
            <input value={prompt} onChange={(event) => setPrompt(event.target.value)} />
          </label>
          <label>
            期待する次の単語
            <input
              value={expectedAnswer}
              onChange={(event) => setExpectedAnswer(event.target.value)}
            />
          </label>
          <div className="actions">
            <button type="button" onClick={fillRandomPrompt}>
              Random Sample
            </button>
            <button type="button" onClick={openSamples}>
              Show Samples
            </button>
            <button type="submit" className="primary" disabled={isStarting || !prompt.trim()}>
              {isStarting ? "Starting..." : "Go"}
            </button>
          </div>
        </form>
      </section>

      {message && <div className="notice">{message}</div>}

      <section className="analysis">
        <header className="statusBar">
          <div>
            <p className="eyebrow">Status</p>
            <h2>
              {status
                ? status.current_step
                : "次の単語を予測する文と、期待する単語を入力して Go ボタンを押してください"}
            </h2>
          </div>
          {status && (
            <div className="prediction">
              <span>AI の予測</span>
              <strong className={status.is_correct === false ? "wrong" : "right"}>
                {status.output ?? "..."}
              </strong>
            </div>
          )}
        </header>

        {status && (
          <>
          <div className="progressTrack" aria-label={`解析進捗 ${progress}%`}>
            <div style={{ width: `${progress}%` }} />
          </div>
          <p className="progressText">
            {status.completed_nodes} / {status.total_nodes ?? "..."} nodes ready
          </p>

          {status.n_layers && status.n_heads ? (
            <ModelGraph status={status} onOpenNode={openNode} />
          ) : (
            <div className="waitingPanel">モデル構造を準備中です</div>
          )}

          {status.state === "failed" && <div className="error">{status.error}</div>}
          </>
        )}
      </section>

      {selected && <ImageModal detail={selected} onClose={() => setSelected(null)} />}
      {isSamplesOpen && (
        <SampleModal
          samples={samples}
          isLoading={isLoadingSamples}
          onSelect={selectSample}
          onClose={() => setIsSamplesOpen(false)}
        />
      )}
    </main>
  );
}

function ModelGraph({
  status,
  onOpenNode,
}: {
  status: JobStatus;
  onOpenNode: (nodeName: string, node: NodeStatus) => void;
}) {
  const layers = Array.from({ length: status.n_layers ?? 0 }, (_, layer) => layer);
  const heads = Array.from({ length: status.n_heads ?? 0 }, (_, head) => head);
  const nLayers = status.n_layers ?? 0;
  const headSpacing = 86;
  const graphWidth = Math.max(1120, heads.length * headSpacing + 240);
  const centerX = graphWidth / 2;
  const outputY = 62;
  const layerGap = 150;
  const headToMlpGap = 72;
  const terminalGap = 112;
  const nodeHeight = 44;
  const headWidth = 72;
  const mainWidth = 120;
  const inputY = outputY + terminalGap + layers.length * layerGap;
  const graphHeight = inputY + 72;

  const graphNodes = useMemo(() => {
    const result: GraphNode[] = [
      {
        name: "Output",
        node: status.nodes.Output,
        x: centerX,
        y: outputY,
        width: mainWidth,
        height: nodeHeight,
      },
    ];

    for (const layer of layers) {
      const visualIndex = nLayers - 1 - layer;
      const mlpY = outputY + terminalGap + visualIndex * layerGap;
      const headY = mlpY + headToMlpGap;
      const headStartX = centerX - ((heads.length - 1) * headSpacing) / 2;

      const mlpName = `MLP${layer}`;
      result.push({
        name: mlpName,
        node: status.nodes[mlpName],
        x: centerX,
        y: mlpY,
        width: mainWidth,
        height: nodeHeight,
      });

      for (const head of heads) {
        const name = `A${layer}.H${head}`;
        result.push({
          name,
          node: status.nodes[name],
          x: headStartX + head * headSpacing,
          y: headY,
          width: headWidth,
          height: nodeHeight,
        });
      }
    }

    result.push({
      name: "Input",
      node: status.nodes.Input,
      x: centerX,
      y: inputY,
      width: mainWidth,
      height: nodeHeight,
    });

    return result;
  }, [centerX, heads, inputY, layers, nLayers, outputY, status.nodes]);

  const nodesByName = useMemo(
    () => new Map(graphNodes.map((node) => [node.name, node])),
    [graphNodes],
  );

  const edges = useMemo(() => {
    const result: Array<[string, string]> = [];
    for (const head of heads) {
      result.push(["Input", `A0.H${head}`]);
    }
    for (const layer of layers) {
      for (const head of heads) {
        result.push([`A${layer}.H${head}`, `MLP${layer}`]);
      }
      if (layer < layers.length - 1) {
        for (const head of heads) {
          result.push([`MLP${layer}`, `A${layer + 1}.H${head}`]);
        }
      }
    }
    if (layers.length > 0) {
      result.push([`MLP${layers.length - 1}`, "Output"]);
    }
    return result;
  }, [heads, layers]);

  return (
    <div className="graphShell">
      <svg
        className="modelSvg"
        viewBox={`0 0 ${graphWidth} ${graphHeight}`}
        role="img"
        aria-label="Transformer model graph"
      >
        <g className="edges">
          {edges.map(([sourceName, targetName]) => {
            const source = nodesByName.get(sourceName);
            const target = nodesByName.get(targetName);
            if (!source || !target) return null;
            const sourceReady = Boolean(source.node?.ready);
            const targetReady = Boolean(target.node?.ready);
            return (
              <path
                key={`${sourceName}-${targetName}`}
                d={curvePath(source, target)}
                stroke={rankColor(source.node?.rank ?? null)}
                className={sourceReady && targetReady ? "edge ready" : "edge pending"}
              />
            );
          })}
        </g>
        <g className="nodes">
          {graphNodes.map((graphNode) => (
            <SvgNode
              key={graphNode.name}
              graphNode={graphNode}
              expectedAnswer={status.expected_answer}
              onOpenNode={onOpenNode}
            />
          ))}
        </g>
      </svg>
    </div>
  );
}

function curvePath(source: GraphNode, target: GraphNode) {
  const x1 = source.x;
  const x2 = target.x;
  const targetIsBelow = target.y > source.y;
  const y1 = source.y + (targetIsBelow ? source.height / 2 : -source.height / 2);
  const y2 = target.y + (targetIsBelow ? -target.height / 2 : target.height / 2);
  const bend = Math.max(32, Math.abs(y2 - y1) * 0.48);
  const direction = targetIsBelow ? 1 : -1;
  return `M ${x1} ${y1} C ${x1} ${y1 + direction * bend}, ${x2} ${y2 - direction * bend}, ${x2} ${y2}`;
}

function SvgNode({
  graphNode,
  expectedAnswer,
  onOpenNode,
}: {
  graphNode: GraphNode;
  expectedAnswer: string;
  onOpenNode: (nodeName: string, node: NodeStatus) => void;
}) {
  const { name, node, x, y, width, height } = graphNode;
  const ready = Boolean(node?.ready);
  const color = rankColor(node?.rank ?? null);
  const fill =
    node?.kind === "attention"
      ? "#ff8d8d"
      : node?.kind === "mlp"
        ? "#f8fff8"
        : node?.kind === "output"
          ? "#ffe0ad"
          : "#d4dae3";
  const rankLabel =
    expectedAnswer && node?.rank
      ? `期待する次の単語「${expectedAnswer}」: ${node.rank} 位`
      : "順位は未計算です";

  function open() {
    if (node?.ready) onOpenNode(name, node);
  }

  function onKeyDown(event: React.KeyboardEvent<SVGGElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  }

  return (
    <g
      className={`svgNode ${ready ? "ready" : "pending"} ${node?.kind ?? ""}`}
      role={ready ? "button" : "img"}
      tabIndex={ready ? 0 : undefined}
      aria-label={ready ? `${name}、${rankLabel}` : `${name} は生成中`}
      style={{ "--rank-color": color } as React.CSSProperties}
      onClick={open}
      onKeyDown={onKeyDown}
    >
      <title>{ready ? rankLabel : `${name} は生成中`}</title>
      <rect
        x={x - width / 2}
        y={y - height / 2}
        width={width}
        height={height}
        rx={8}
        fill={fill}
        stroke={color}
      />
      <text x={x} y={y + 7} textAnchor="middle">
        {name}
      </text>
    </g>
  );
}

function ImageModal({ detail, onClose }: { detail: NodeDetail; onClose: () => void }) {
  return (
    <div className="modal" onClick={onClose}>
      <div className="modalBody" onClick={(event) => event.stopPropagation()}>
        <header>
          <div>
            <p className="eyebrow">{detail.node}</p>
            <h2>{detail.rank ? `rank ${detail.rank}` : "Detail"}</h2>
          </div>
          <button onClick={onClose} aria-label="閉じる">
            x
          </button>
        </header>
        <div className={detail.attention_image && detail.logits_image ? "imageGrid" : "imageGrid single"}>
          {detail.attention_image && (
            <figure>
              <figcaption>注意パターン</figcaption>
              <img src={`data:image/png;base64,${detail.attention_image}`} alt={`${detail.node} attention`} />
            </figure>
          )}
          {detail.logits_image && (
            <figure>
              <figcaption>予測ランキング</figcaption>
              <img src={`data:image/png;base64,${detail.logits_image}`} alt={`${detail.node} logits`} />
            </figure>
          )}
        </div>
      </div>
    </div>
  );
}

function SampleModal({
  samples,
  isLoading,
  onSelect,
  onClose,
}: {
  samples: PromptSample[];
  isLoading: boolean;
  onSelect: (sample: PromptSample) => void;
  onClose: () => void;
}) {
  return (
    <div className="modal" onClick={onClose}>
      <div className="modalBody sampleModalBody" onClick={(event) => event.stopPropagation()}>
        <header>
          <div>
            <p className="eyebrow">Samples</p>
            <h2>サンプルを選択</h2>
          </div>
          <button onClick={onClose} aria-label="閉じる">
            x
          </button>
        </header>
        {isLoading ? (
          <div className="waitingPanel compact">サンプル一覧を読み込み中です</div>
        ) : (
          <div className="sampleList">
            {samples.map((sample) => (
              <button
                className="sampleItem"
                key={`${sample.prompt}-${sample.expected_answer}`}
                onClick={() => onSelect(sample)}
              >
                <span className="sampleSubject">{sample.subject}</span>
                <span className="samplePrompt">{sample.prompt}</span>
                <span className="sampleMeta">
                  answer: {sample.expected_answer} / {sample.keywords}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
