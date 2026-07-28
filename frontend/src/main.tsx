import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
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
  attention: AttentionData | null;
  logits: LogitsData | null;
};

type AttentionData = {
  tokens: string[];
  values: number[][];
};

type LogitsData = {
  tokens: string[];
  values: number[];
};

type PromptSample = {
  prompt: string;
  subject: string;
  expected_answer: string;
  keywords: string;
};

type GraphNode = {
  name: string;
  kind: NodeKind;
  x: number;
  y: number;
  width: number;
  height: number;
  fontSize: number;
};

type GraphLayout = {
  width: number;
  height: number;
  nodes: GraphNode[];
  edges: Array<[string, string]>;
  strokeWidth: number;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

const clamp = (value: number, min: number, max: number) =>
  Math.max(min, Math.min(max, value));

// Rank -> color. Vivid green means the expected token ranks near the top at
// this node; it fades to a cool slate-blue when the token is far down (or
// unknown). Kept saturated so the node borders read clearly on the dark stage.
function rankColor(rank: number | null, vocabThreshold = 5025) {
  const dim: [number, number, number] = [96, 132, 156];
  const bright: [number, number, number] = [40, 224, 122];
  if (rank === null || rank >= vocabThreshold) {
    return `rgb(${dim[0]}, ${dim[1]}, ${dim[2]})`;
  }
  const t = clamp(
    Math.log(rank + 1e-8) / Math.log(vocabThreshold + 1e-8),
    0,
    1,
  );
  const c = bright.map((b, i) => Math.round(b + (dim[i] - b) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

const NODE_FILL: Record<NodeKind, string> = {
  input: "#d3dade",
  attention: "#dceaf4",
  mlp: "#e0ebe0",
  output: "#f2e6cf",
};

// Observe an element's rendered size so the graph can fill it exactly.
function useElementSize<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [size, setSize] = useState({ width: 0, height: 0 });

  useEffect(() => {
    const element = ref.current;
    if (!element) return;
    const observer = new ResizeObserver((entries) => {
      const rect = entries[0].contentRect;
      setSize({ width: rect.width, height: rect.height });
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, size] as const;
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
  // Show the explanation on load; re-openable from the title.
  const [isAboutOpen, setIsAboutOpen] = useState(true);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let intervalId: number | undefined;

    async function poll() {
      const response = await fetch(`${API_BASE}/api/jobs/${jobId}`);
      if (!response.ok) return;
      const nextStatus = (await response.json()) as JobStatus;
      if (cancelled) return;
      setStatus(nextStatus);
      // The job is immutable once it reaches a terminal state, so stop polling.
      if (nextStatus.state === "completed" || nextStatus.state === "failed") {
        if (intervalId !== undefined) window.clearInterval(intervalId);
      }
    }

    poll();
    intervalId = window.setInterval(poll, 1200);
    return () => {
      cancelled = true;
      if (intervalId !== undefined) window.clearInterval(intervalId);
    };
  }, [jobId]);

  useEffect(() => {
    if (!message) return;
    const timeoutId = window.setTimeout(() => setMessage(null), 4000);
    return () => window.clearTimeout(timeoutId);
  }, [message]);

  async function startAnalysis(event: React.FormEvent) {
    event.preventDefault();
    const normalizedPrompt = prompt.trim();
    const normalizedExpectedAnswer = expectedAnswer.trim();
    setPrompt(normalizedPrompt);
    setExpectedAnswer(normalizedExpectedAnswer);
    setMessage(null);
    setSelected(null);
    setStatus(null);
    setIsStarting(true);
    try {
      const response = await fetch(`${API_BASE}/api/analyze`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: normalizedPrompt,
          expected_answer: normalizedExpectedAnswer,
        }),
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
    if (!jobId || !node.ready || node.kind === "input") return;
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

  const hasModel = Boolean(status?.n_layers && status?.n_heads);
  const predictionClass =
    status?.is_correct === true ? "right" : status?.is_correct === false ? "wrong" : "";

  // Give the graph a comfortable minimum height (~one row per level). On a short
  // landscape monitor this exceeds the viewport, so the page scrolls with roomy
  // spacing; on a tall portrait monitor it fits and the stage simply fills.
  const stageStyle =
    hasModel && status?.n_layers
      ? { minHeight: (2 + 2 * status.n_layers) * 52 + 40 }
      : undefined;

  return (
    <div className="app">
      <header className="topbar">
        <div className="topbar-top">
          <div className="brand">
            <h1 className="brand-name">Visualize LLM Demo</h1>
            <p className="brand-desc">次の単語を予測する仕組み</p>
          </div>

          <div className="topbar-tools">
            <button
              type="button"
              className="help"
              onClick={() => setIsAboutOpen(true)}
              title="このデモの説明を表示"
              aria-label="このデモの説明を表示"
            >
              ?
            </button>
            <button type="button" className="ghost" onClick={fillRandomPrompt}>
              例文からランダムに選ぶ
            </button>
            <button type="button" className="ghost" onClick={openSamples}>
              例文を見て選ぶ
            </button>
          </div>
        </div>

        <form className="controls" onSubmit={startAnalysis}>
          <div className="field grow">
            <label htmlFor="prompt-input">入力</label>
            <input
              id="prompt-input"
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              autoComplete="off"
            />
          </div>
          <div className="field narrow">
            <label htmlFor="answer-input">期待する次の単語</label>
            <input
              id="answer-input"
              value={expectedAnswer}
              onChange={(event) => setExpectedAnswer(event.target.value)}
              autoComplete="off"
            />
          </div>
          <div className="actions">
            <button type="submit" className="primary" disabled={isStarting || !prompt.trim()}>
              {isStarting ? "Starting…" : "Go"}
            </button>
          </div>
        </form>
      </header>

      <div className="statusStrip">
        <p className="status-step">
          {status
            ? status.current_step
            : "文と期待する次の単語を入力して Go を押してください"}
        </p>
        {status && (
          <p className="prediction">
            <span className="prediction-label">AI の予測</span>
            <strong className={predictionClass}>{status.output ?? "…"}</strong>
          </p>
        )}
      </div>

      <main className="stage" style={stageStyle}>
        {hasModel && status ? (
          <ModelGraph status={status} onOpenNode={openNode} />
        ) : (
          <div className="graphShell">
            <div className="stagePlaceholder">
              <p className="big">
                {status ? "モデル構造を準備中です" : "INPUT から OUTPUT までの流れを可視化します"}
              </p>
              <p className="sub">
                {status
                  ? status.current_step
                  : "Transformer の各層が、期待する単語をどれだけ予測できているかを表示します"}
              </p>
            </div>
          </div>
        )}
        {status?.state === "failed" && status.error && (
          <div className="stageError">{status.error}</div>
        )}
      </main>

      {message && <div className="toast">{message}</div>}

      {selected && (
        <DetailModal
          detail={selected}
          expected={status?.expected_answer ?? ""}
          onClose={() => setSelected(null)}
        />
      )}
      {isSamplesOpen && (
        <SampleModal
          samples={samples}
          isLoading={isLoadingSamples}
          onSelect={selectSample}
          onClose={() => setIsSamplesOpen(false)}
        />
      )}
      {isAboutOpen && <AboutModal onClose={() => setIsAboutOpen(false)} />}
    </div>
  );
}

function ModalFrame({
  eyebrow,
  title,
  bodyClassName = "",
  headerExtra,
  children,
  onClose,
}: {
  eyebrow: string;
  title: React.ReactNode;
  bodyClassName?: string;
  headerExtra?: React.ReactNode;
  children: React.ReactNode;
  onClose: () => void;
}) {
  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="modal" onClick={onClose}>
      <div
        className={`modalBody ${bodyClassName}`.trim()}
        onClick={(event) => event.stopPropagation()}
      >
        <header>
          <div>
            <p className="eyebrow">{eyebrow}</p>
            <h2>{title}</h2>
            {headerExtra}
          </div>
          <button className="ghost closeButton" onClick={onClose} aria-label="閉じる">
            ✕
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}

function AboutModal({ onClose }: { onClose: () => void }) {
  return (
    <ModalFrame
      eyebrow="About"
      title="このデモについて"
      bodyClassName="aboutModalBody"
      onClose={onClose}
    >
      <div className="aboutContent">
        <section>
          <h3>👀 何を見るデモ？</h3>
          <p>
            <strong>AI（大規模言語モデル）が次の単語を予測する仕組み</strong>を可視化します。
            ChatGPT や翻訳 AI は「文章の次の単語を予測する」ことを繰り返して文章を作っています。
            たとえば「雲の切れ間から、光が」→「<span className="accentWord">差す</span>」のように、
            続きの単語を予測します。
            その予測が AI の内部でどう組み立てられるかを、層ごとに覗けます。
          </p>
        </section>

        <section>
          <h3>📝 使い方</h3>
          <ul>
            <li>
              文章を<strong>途中まで</strong>入力（例: Sendai is located in the country of）
            </li>
            <li>
              AI に予測させたい<strong>次の単語</strong>を入力（例: Japan）
            </li>
            <li>
              <strong>Go</strong> で解析開始。AI 内部の予測の流れが図として表示されます
            </li>
            <li>
              それぞれの箱を<strong>クリック</strong>すると、詳細（注意パターン・予測ランキング）が見られます
            </li>
          </ul>
          <p>
            💡 迷ったら <strong>例文からランダムに選ぶ</strong> / <strong>例文を見て選ぶ</strong> でサンプル文章を入力できます
          </p>
        </section>

        <section>
          <h3>🔍 図の見方</h3>
          <ul>
            <li>
              <strong>Input</strong>: 入力文を AI が処理できる形に加工する部分
            </li>
            <li>
              <strong>A0.H0</strong> など（Attention）: どの単語に注目するかを決める場所
            </li>
            <li>
              <strong>MLP0</strong> など: 注目した情報から「次の単語のヒント」を作る場所
            </li>
            <li>
              <strong>Output</strong>: 🎯 最終的な予測（次の単語）を決める部分
            </li>
          </ul>
        </section>

        <section>
          <h3>🎨 色の意味</h3>
          <ul>
            <li>
              箱の<span className="accent-green">枠線・接続線が緑に近い</span>ほど、その地点で期待する単語を
              <strong>上位で予測</strong>している（正解に近い）
            </li>
            <li>グレーに近いほど順位が低い（5000 位以下はグレーで固定）</li>
          </ul>
        </section>
      </div>
    </ModalFrame>
  );
}

// Build a layout that fills the given box: layers are distributed across the
// full height (INPUT top, OUTPUT bottom — all visible) and attention heads
// spread across the full width. Works for any container aspect ratio, so a
// landscape monitor uses the width and a pivoted monitor uses the height.
function computeLayout(
  nLayers: number,
  nHeads: number,
  width: number,
  height: number,
): GraphLayout | null {
  if (!width || !height || nLayers <= 0 || nHeads <= 0) return null;

  const padX = clamp(width * 0.03, 24, 96);
  const padTop = clamp(height * 0.05, 18, 56);
  // Reserve room at the bottom for the legend overlay so the Output node,
  // which sits centered on the last row, never collides with it.
  const padBottom = padTop + 44;

  // Ordered vertical levels, top -> bottom.
  type Level =
    | { kind: "output" | "input"; name: string }
    | { kind: "mlp"; name: string; layer: number }
    | { kind: "attn"; layer: number };

  const levels: Level[] = [{ kind: "input", name: "Input" }];
  for (let layer = 0; layer < nLayers; layer += 1) {
    levels.push({ kind: "attn", layer });
    levels.push({ kind: "mlp", name: `MLP${layer}`, layer });
  }
  levels.push({ kind: "output", name: "Output" });

  // Even spacing between every level, so each attention row sits exactly
  // halfway between the MLP below it (toward Output) and the MLP above it
  // (toward Input).
  const usableHeight = height - padTop - padBottom;
  const rowGap = usableHeight / (levels.length - 1);

  const yByLevel: number[] = [];
  for (let i = 0; i < levels.length; i += 1) {
    yByLevel.push(padTop + i * rowGap);
  }

  const nodeHeight = clamp(rowGap * 0.6, 14, 40);
  const centerX = width / 2;
  const usableWidth = width - 2 * padX;
  const headSlot = usableWidth / nHeads;
  const headWidth = clamp(headSlot * 0.82, 20, 128);
  const mainWidth = clamp(headWidth * 1.6, 68, 190);
  const mainFont = clamp(nodeHeight * 0.44, 9, 16);
  const headFont = clamp(Math.min(nodeHeight * 0.44, headWidth * 0.24), 7, 14);

  const nodes: GraphNode[] = [];
  levels.forEach((level, index) => {
    const y = yByLevel[index];
    if (level.kind === "attn") {
      for (let head = 0; head < nHeads; head += 1) {
        nodes.push({
          name: `A${level.layer}.H${head}`,
          kind: "attention",
          x: padX + headSlot * head + headSlot / 2,
          y,
          width: headWidth,
          height: nodeHeight,
          fontSize: headFont,
        });
      }
    } else {
      nodes.push({
        name: level.name,
        kind: level.kind,
        x: centerX,
        y,
        width: mainWidth,
        height: nodeHeight,
        fontSize: mainFont,
      });
    }
  });

  const edges: Array<[string, string]> = [];
  for (let head = 0; head < nHeads; head += 1) edges.push(["Input", `A0.H${head}`]);
  for (let layer = 0; layer < nLayers; layer += 1) {
    for (let head = 0; head < nHeads; head += 1) {
      edges.push([`A${layer}.H${head}`, `MLP${layer}`]);
    }
    if (layer < nLayers - 1) {
      for (let head = 0; head < nHeads; head += 1) {
        edges.push([`MLP${layer}`, `A${layer + 1}.H${head}`]);
      }
    }
  }
  if (nLayers > 0) edges.push([`MLP${nLayers - 1}`, "Output"]);

  return {
    width,
    height,
    nodes,
    edges,
    strokeWidth: clamp(nodeHeight * 0.08, 1.5, 3),
  };
}

function ModelGraph({
  status,
  onOpenNode,
}: {
  status: JobStatus;
  onOpenNode: (nodeName: string, node: NodeStatus) => void;
}) {
  const [ref, size] = useElementSize<HTMLDivElement>();
  const [tip, setTip] = useState<TipData | null>(null);
  const nLayers = status.n_layers ?? 0;
  const nHeads = status.n_heads ?? 0;

  const showTip = useCallback((info: TipInfo, event: React.MouseEvent) => {
    setTip({ ...info, x: event.clientX, y: event.clientY });
  }, []);
  const hideTip = useCallback(() => setTip(null), []);

  // Positions depend only on model shape and container size, not on which
  // nodes are ready yet, so live polling updates never re-flow the graph.
  const layout = useMemo(
    () => computeLayout(nLayers, nHeads, size.width, size.height),
    [nLayers, nHeads, size.width, size.height],
  );

  const nodesByName = useMemo(() => {
    if (!layout) return new Map<string, GraphNode>();
    return new Map(layout.nodes.map((node) => [node.name, node]));
  }, [layout]);

  return (
    <div className="graphShell" ref={ref}>
      {layout && (
        <svg
          className="modelSvg"
          viewBox={`0 0 ${layout.width} ${layout.height}`}
          preserveAspectRatio="none"
          role="img"
          aria-label="Transformer model graph"
        >
          <g className="edges">
            {layout.edges.map(([sourceName, targetName]) => {
              const source = nodesByName.get(sourceName);
              const target = nodesByName.get(targetName);
              if (!source || !target) return null;
              const sourceNode = status.nodes[sourceName];
              const targetNode = status.nodes[targetName];
              const ready = Boolean(sourceNode?.ready && targetNode?.ready);
              return (
                <path
                  key={`${sourceName}-${targetName}`}
                  d={curvePath(source, target)}
                  stroke={rankColor(sourceNode?.rank ?? null)}
                  strokeWidth={layout.strokeWidth}
                  className={ready ? "edge ready" : "edge pending"}
                />
              );
            })}
          </g>
          <g className="nodes">
            {layout.nodes.map((graphNode) => (
              <SvgNode
                key={graphNode.name}
                graphNode={graphNode}
                node={status.nodes[graphNode.name]}
                expectedAnswer={status.expected_answer}
                strokeWidth={layout.strokeWidth * 2.6}
                onOpenNode={onOpenNode}
                onShowTip={showTip}
                onHideTip={hideTip}
              />
            ))}
          </g>
        </svg>
      )}

      {tip && <NodeTooltip tip={tip} />}

      <div className="legend">
        <span className="legend-item">
          <span className="legend-gradient" />
          枠線・接続線が緑に近いほど、その地点で期待する単語の順位が高い (5000位以下はグレー表示)
        </span>
      </div>
    </div>
  );
}

type TipInfo = {
  name: string;
  expected: string;
  rank: number | null;
  ready: boolean;
};

type TipData = TipInfo & { x: number; y: number };

// Cursor-anchored tooltip, flipping away from the right/bottom viewport edges.
// Fixed-positioned so it escapes overflow/scroll clipping and modal stacking.
function Tooltip({
  x,
  y,
  children,
}: {
  x: number;
  y: number;
  children: React.ReactNode;
}) {
  const flipX = x > window.innerWidth - 320;
  const flipY = y > window.innerHeight - 110;
  const style: React.CSSProperties = {
    left: x,
    top: y,
    transform: `translate(${flipX ? "calc(-100% - 16px)" : "16px"}, ${
      flipY ? "calc(-100% - 12px)" : "18px"
    })`,
  };
  return (
    <div className="tooltip" style={style}>
      {children}
    </div>
  );
}

function NodeTooltip({ tip }: { tip: TipData }) {
  return (
    <Tooltip x={tip.x} y={tip.y}>
      <span className="tip-name">{tip.name}</span>
      <span className="tip-detail">
        {!tip.ready ? (
          "生成中…"
        ) : tip.rank != null ? (
          <>
            期待する単語「{tip.expected}」: <span className="tip-rank">{tip.rank}</span> 位
          </>
        ) : (
          "順位は未計算"
        )}
      </span>
    </Tooltip>
  );
}

function curvePath(source: GraphNode, target: GraphNode) {
  const x1 = source.x;
  const x2 = target.x;
  const targetIsBelow = target.y > source.y;
  const y1 = source.y + (targetIsBelow ? source.height / 2 : -source.height / 2);
  const y2 = target.y + (targetIsBelow ? -target.height / 2 : target.height / 2);
  const bend = Math.max(24, Math.abs(y2 - y1) * 0.48);
  const direction = targetIsBelow ? 1 : -1;
  return `M ${x1} ${y1} C ${x1} ${y1 + direction * bend}, ${x2} ${y2 - direction * bend}, ${x2} ${y2}`;
}

const SvgNode = React.memo(function SvgNode({
  graphNode,
  node,
  expectedAnswer,
  strokeWidth,
  onOpenNode,
  onShowTip,
  onHideTip,
}: {
  graphNode: GraphNode;
  node: NodeStatus | undefined;
  expectedAnswer: string;
  strokeWidth: number;
  onOpenNode: (nodeName: string, node: NodeStatus) => void;
  onShowTip: (info: TipInfo, event: React.MouseEvent) => void;
  onHideTip: () => void;
}) {
  const { name, kind, x, y, width, height, fontSize } = graphNode;
  const ready = Boolean(node?.ready);
  const canOpen = ready && kind !== "input";
  const color = rankColor(node?.rank ?? null);
  const fill = NODE_FILL[kind];
  const rankLabel =
    expectedAnswer && node?.rank
      ? `期待する次の単語「${expectedAnswer}」: ${node.rank} 位`
      : "順位は未計算です";

  function open() {
    if (node && canOpen) onOpenNode(name, node);
  }

  function onKeyDown(event: React.KeyboardEvent<SVGGElement>) {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  }

  function onEnter(event: React.MouseEvent) {
    onShowTip({ name, expected: expectedAnswer, rank: node?.rank ?? null, ready }, event);
  }

  return (
    <g
      className={`svgNode ${ready ? "ready" : "pending"} ${
        canOpen ? "interactive" : "static"
      } ${kind}`}
      role={canOpen ? "button" : "img"}
      tabIndex={canOpen ? 0 : undefined}
      aria-label={ready ? `${name}、${rankLabel}` : `${name} は生成中`}
      style={{ "--rank-color": color } as React.CSSProperties}
      onClick={open}
      onKeyDown={onKeyDown}
      onMouseEnter={onEnter}
      onMouseLeave={onHideTip}
    >
      <rect
        x={x - width / 2}
        y={y - height / 2}
        width={width}
        height={height}
        rx={Math.min(8, height / 2)}
        fill={fill}
        stroke={color}
        strokeWidth={strokeWidth}
      />
      <text
        x={x}
        y={y}
        textAnchor="middle"
        dominantBaseline="central"
        fontSize={fontSize}
      >
        {name}
      </text>
    </g>
  );
});

function DetailModal({
  detail,
  expected,
  onClose,
}: {
  detail: NodeDetail;
  expected: string;
  onClose: () => void;
}) {
  return (
    <ModalFrame
      eyebrow={detail.node}
      title={
        detail.rank ? `期待する単語「${expected}」: ${detail.rank} 位` : "ノードの詳細"
      }
      onClose={onClose}
    >
      <div className={detail.attention && detail.logits ? "detailGrid" : "detailGrid single"}>
        {detail.attention && (
          <figure>
            <figcaption>注意パターン</figcaption>
            <AttentionHeatmap data={detail.attention} />
          </figure>
        )}
        {detail.logits && (
          <figure>
            <figcaption>予測ランキング</figcaption>
            <LogitsRanking data={detail.logits} />
          </figure>
        )}
      </div>
    </ModalFrame>
  );
}

function AttentionHeatmap({ data }: { data: AttentionData }) {
  const cellSize = 28;
  const labelSize = 118;
  const size = data.tokens.length * cellSize;
  const maxValue = Math.max(...data.values.flat(), 1);
  const [tip, setTip] = useState<{ row: number; col: number; value: number; x: number; y: number } | null>(
    null,
  );

  return (
    <div className="heatmapScroll">
      <svg
        className="attentionSvg"
        viewBox={`0 0 ${labelSize + size} ${labelSize + size}`}
        role="img"
        aria-label="Attention heatmap"
      >
        {/* Only the top-left corner is white; everything else keeps the panel
            background, so the masked area above the diagonal blends into the
            label gutters instead of reading as stripes. */}
        <rect x={0} y={0} width={labelSize} height={labelSize} fill="#ffffff" />
        {data.tokens.map((token, index) => {
          const colCenter = labelSize + index * cellSize + cellSize / 2;
          return (
          <g key={`label-${index}`} className="heatmapLabel">
            <text
              x={colCenter}
              y={labelSize - 8}
              textAnchor="start"
              dominantBaseline="central"
              transform={`rotate(270 ${colCenter} ${labelSize - 8})`}
            >
              {token}
            </text>
            <text
              x={labelSize - 10}
              y={labelSize + index * cellSize + cellSize / 2 + 4}
              textAnchor="end"
            >
              {token}
            </text>
          </g>
          );
        })}
        {data.values.flatMap((row, rowIndex) =>
          row.map((value, columnIndex) => {
            // Above the diagonal is masked by causal attention (always 0.0):
            // draw nothing, no tooltip / no hover.
            if (columnIndex > rowIndex) {
              return null;
            }
            const x = labelSize + columnIndex * cellSize;
            const y = labelSize + rowIndex * cellSize;
            const intensity = Math.max(0, Math.min(1, value / maxValue));
            return (
              <rect
                key={`${rowIndex}-${columnIndex}`}
                x={x}
                y={y}
                width={cellSize}
                height={cellSize}
                fill={attentionColor(intensity)}
                stroke="#ffffff"
                strokeWidth="1"
                onMouseEnter={(event) =>
                  setTip({ row: rowIndex, col: columnIndex, value, x: event.clientX, y: event.clientY })
                }
                onMouseLeave={() => setTip(null)}
              />
            );
          }),
        )}
      </svg>
      {tip && (
        <Tooltip x={tip.x} y={tip.y}>
          <span className="tip-name">
            {data.tokens[tip.row]} → {data.tokens[tip.col]}
          </span>
          <span className="tip-detail">
            attention: <span className="tip-rank">{tip.value.toFixed(3)}</span>
          </span>
        </Tooltip>
      )}
    </div>
  );
}

// Sequential ramp along the primary hue: primary-50 (low) -> primary-700 (high).
function attentionColor(intensity: number) {
  const from: [number, number, number] = [241, 247, 253];
  const to: [number, number, number] = [39, 79, 124];
  const t = Math.max(0, Math.min(1, intensity));
  const c = from.map((f, i) => Math.round(f + (to[i] - f) * t));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

function LogitsRanking({ data }: { data: LogitsData }) {
  const minimum = Math.min(...data.values);
  const maximum = Math.max(...data.values);
  const range = maximum - minimum || 1;

  return (
    <div className="logitsList">
      {data.tokens.map((token, index) => {
        const width = 18 + ((data.values[index] - minimum) / range) * 82;
        return (
          <div className="logitRow" key={`${token}-${index}`}>
            <span className="logitToken">{token}</span>
            <div className="logitBarTrack">
              <div className="logitBar" style={{ width: `${width}%` }} />
            </div>
            <span className="logitValue">{data.values[index].toFixed(2)}</span>
          </div>
        );
      })}
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
    <ModalFrame
      eyebrow="Samples"
      title="サンプルを選択"
      bodyClassName="sampleModalBody"
      headerExtra={
        <p className="modal-note">
          <span className="accentWord">この色のテキスト</span> が期待する次の単語です
        </p>
      }
      onClose={onClose}
    >
      {isLoading ? (
        <div className="waitingPanel">サンプル一覧を読み込み中です</div>
      ) : (
        <div className="sampleList">
          {samples.map((sample) => (
            <button
              className="sampleItem"
              key={`${sample.prompt}-${sample.expected_answer}`}
              onClick={() => onSelect(sample)}
            >
              <span className="samplePrompt">
                {sample.prompt} <span className="accentWord">{sample.expected_answer}</span>
              </span>
            </button>
          ))}
        </div>
      )}
    </ModalFrame>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
