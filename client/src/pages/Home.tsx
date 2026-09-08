import {
  Activity,
  ArrowDown,
  ArrowRight,
  Atom,
  Beaker,
  BrainCircuit,
  Check,
  ChevronDown,
  ChevronUp,
  CircleHelp,
  Clipboard,
  Code2,
  Database,
  ExternalLink,
  Eye,
  FlaskConical,
  Gauge,
  GitBranch,
  Info,
  Layers3,
  Lightbulb,
  Link2,
  Network,
  Play,
  RotateCcw,
  Sparkles,
  TimerReset,
  TriangleAlert,
  X,
  Zap,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  EXAMPLE_COPY,
  MAP_COPY,
  type AttentionRun,
  type ExampleName,
  type FeatureMapName,
  formatCount,
  formatVector,
  complexityPoints,
  matrixMinMax,
  normalizeForColor,
  runAttention,
  vectorBarValues,
} from "@/lib/attention";
import type { Matrix } from "@/lib/attention";

const accent = "#59d8ff";
const warm = "#ffb86c";
const sectionIds = ["why", "softmax", "linear", "lab", "failure", "bdh", "takeaway"];

function cn(...classes: Array<string | false | null | undefined>) {
  return classes.filter(Boolean).join(" ");
}

function SectionLabel({ index, eyebrow, title, description }: { index: string; eyebrow: string; title: string; description: string }) {
  return (
    <div className="section-heading">
      <div className="section-index"><span>{index}</span><span className="section-rule" /></div>
      <p className="eyebrow">{eyebrow}</p>
      <h2>{title}</h2>
      <p className="section-lede">{description}</p>
    </div>
  );
}

function Panel({ children, className = "", tint = "" }: { children: React.ReactNode; className?: string; tint?: string }) {
  return <div className={cn("lab-panel", tint, className)}>{children}</div>;
}

function Tag({ children, tone = "blue" }: { children: React.ReactNode; tone?: "blue" | "warm" | "muted" | "green" }) {
  return <span className={cn("tag", `tag-${tone}`)}>{children}</span>;
}

function InfoNote({ icon: Icon = Info, children, tone = "blue" }: { icon?: typeof Info; children: React.ReactNode; tone?: "blue" | "warm" }) {
  return <div className={cn("info-note", tone === "warm" && "info-note-warm")}><Icon size={15} /> <span>{children}</span></div>;
}

function Stat({ label, value, suffix, tone = "blue" }: { label: string; value: string; suffix?: string; tone?: "blue" | "warm" | "muted" }) {
  return <div className={cn("stat", `stat-${tone}`)}><span className="stat-label">{label}</span><strong>{value}<small>{suffix}</small></strong></div>;
}

function TokenStrip({ tokens, selected, onSelect }: { tokens: string[]; selected: number; onSelect: (index: number) => void }) {
  return (
    <div className="token-strip" aria-label="Token sequence">
      {tokens.map((token, index) => (
        <button key={`${token}-${index}`} className={cn("token-chip", selected === index && "token-chip-selected")} onClick={() => onSelect(index)} aria-label={`Select token ${token} at position ${index + 1}`}>
          <span>{String(index + 1).padStart(2, "0")}</span>{token}
        </button>
      ))}
    </div>
  );
}

function MatrixCard({ label, matrix, tokens, selectedRow, kind = "exact" }: { label: string; matrix: Matrix; tokens: string[]; selectedRow: number; kind?: "exact" | "linear" | "difference" }) {
  const { min, max } = matrixMinMax(matrix);
  return (
    <div className={cn("matrix-card", `matrix-${kind}`)}>
      <div className="matrix-card-head"><span>{label}</span><span className="matrix-dims">{matrix.length} × {matrix[0]?.length ?? 0}</span></div>
      <div className="matrix-grid" style={{ gridTemplateColumns: `repeat(${matrix[0]?.length ?? 1}, minmax(0, 1fr))` }}>
        {matrix.flatMap((row, rowIndex) => row.map((value, columnIndex) => {
          const normalized = normalizeForColor(value, min, max);
          const isSelected = kind !== "difference" && rowIndex === selectedRow;
          const isDiff = kind === "difference";
          return <div key={`${rowIndex}-${columnIndex}`} className={cn("matrix-cell", isSelected && "matrix-cell-selected", isDiff && "matrix-cell-diff")} title={`${tokens[rowIndex] ?? "row"} → ${tokens[columnIndex] ?? "column"}: ${value.toFixed(3)}`} style={{ "--cell-alpha": `${0.12 + normalized * 0.78}`, "--cell-hue": isDiff ? `${26 + normalized * 20}` : `${190 + normalized * 35}` } as React.CSSProperties}>{Math.abs(value) < 0.01 ? "·" : value.toFixed(2)}</div>;
        }))}
      </div>
      <div className="matrix-axis"><span>query rows</span><span>key / feature columns</span></div>
    </div>
  );
}

function VectorBars({ vector, label, tone = "blue" }: { vector: number[]; label: string; tone?: "blue" | "warm" | "green" }) {
  return (
    <div className="vector-block">
      <div className="mini-label"><span>{label}</span><span className="mono">{vector.length} dims</span></div>
      <div className="vector-bars" aria-label={`${label} values`}>
        {vectorBarValues(vector).map(({ value, ratio }, index) => <div key={index} className="vector-column"><div className={cn("vector-bar", `vector-${tone}`)} style={{ height: `${Math.max(5, ratio * 56)}px`, opacity: 0.35 + ratio * 0.65 }} /><span>{value.toFixed(2)}</span></div>)}
      </div>
    </div>
  );
}

function FlowStep({ label, formula, active = false }: { label: string; formula?: string; active?: boolean }) {
  return <div className={cn("flow-step", active && "flow-step-active")}><span>{label}</span>{formula && <b>{formula}</b>}</div>;
}

function ComplexityChart({ points, currentLength }: { points: Array<{ length: number; exact: number; linear: number }>; currentLength: number }) {
  const width = 720; const height = 220; const pad = { left: 48, right: 18, top: 18, bottom: 34 };
  const max = Math.max(...points.flatMap((point) => [point.exact, point.linear]));
  const x = (length: number) => pad.left + ((length - points[0].length) / (points[points.length - 1].length - points[0].length)) * (width - pad.left - pad.right);
  const y = (value: number) => pad.top + (1 - value / max) * (height - pad.top - pad.bottom);
  const path = (key: "exact" | "linear") => points.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.length).toFixed(1)},${y(point[key]).toFixed(1)}`).join(" ");
  const currentX = x(Math.min(points[points.length - 1].length, Math.max(points[0].length, currentLength)));
  return <div className="chart-wrap"><svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Operation count grows quadratically for exact attention and linearly for the kernelized implementation"><defs><linearGradient id="exactFill" x1="0" x2="0" y1="0" y2="1"><stop offset="0" stopColor="#ffb86c" stopOpacity=".24" /><stop offset="1" stopColor="#ffb86c" stopOpacity="0" /></linearGradient></defs>{[0, .25, .5, .75, 1].map((tick) => <line key={tick} x1={pad.left} x2={width - pad.right} y1={y(max * tick)} y2={y(max * tick)} className="chart-gridline" />)}<path d={`${path("exact")} L${x(points[points.length - 1].length)},${height - pad.bottom} L${x(points[0].length)},${height - pad.bottom} Z`} fill="url(#exactFill)" /><path d={path("exact")} className="chart-line chart-line-exact" /><path d={path("linear")} className="chart-line chart-line-linear" /><line x1={currentX} x2={currentX} y1={pad.top} y2={height - pad.bottom} className="chart-current" />{points.map((point) => <g key={point.length}><circle cx={x(point.length)} cy={y(point.exact)} r="3" className="chart-dot chart-dot-exact" /><circle cx={x(point.length)} cy={y(point.linear)} r="3" className="chart-dot chart-dot-linear" /><text x={x(point.length)} y={height - 12} textAnchor="middle" className="chart-axis-text">{point.length}</text></g>)}<text x={12} y={22} className="chart-axis-title">work</text><text x={width - 13} y={height - 12} textAnchor="end" className="chart-axis-title">sequence length →</text></svg><div className="chart-legend"><span><i className="legend-dot legend-exact" /> Exact Softmax · O(N²)</span><span><i className="legend-dot legend-linear" /> Kernelized · O(N)</span><span className="chart-current-key"><i /> current N = {currentLength}</span></div></div>;
}

function BdhNetwork({ mode, pulse }: { mode: "dense" | "sparse"; pulse: number }) {
  const nodes = [{ x: 35, y: 50 }, { x: 92, y: 28 }, { x: 92, y: 72 }, { x: 150, y: 42 }, { x: 150, y: 83 }, { x: 208, y: 26 }, { x: 208, y: 65 }, { x: 266, y: 48 }];
  type Edge = [{ x: number; y: number }, { x: number; y: number }, string];
  const edges: Edge[] = mode === "dense" ? nodes.flatMap((node, i) => nodes.slice(i + 1, i + 4).map((target, j) => [node, target, `${i}-${j}`] as Edge)) : [[nodes[0], nodes[1], "01"], [nodes[0], nodes[2], "02"], [nodes[1], nodes[3], "13"], [nodes[2], nodes[4], "24"], [nodes[3], nodes[5], "35"], [nodes[4], nodes[6], "46"], [nodes[5], nodes[7], "57"], [nodes[6], nodes[7], "67"]];
  return <svg className="network-svg" viewBox="0 0 300 112" role="img" aria-label={`${mode === "dense" ? "Dense conventional" : "Sparse local"} interaction conceptual visualization`}><g className="network-edges">{edges.map(([from, to, key]) => <line key={key} x1={from.x} y1={from.y} x2={to.x} y2={to.y} className={cn("network-edge", mode === "sparse" && "network-edge-sparse")} style={{ opacity: mode === "dense" ? .18 : .55 }} />)}</g><g>{nodes.map((node, index) => <g key={index} className={cn("network-node", (index + pulse) % 5 === 0 && "network-node-pulse")}><circle cx={node.x} cy={node.y} r={mode === "sparse" && index % 3 === 0 ? 7 : 5} /><text x={node.x} y={node.y + 20} textAnchor="middle">{mode === "sparse" && index % 3 === 0 ? "active" : ""}</text></g>)}</g></svg>;
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: string }) {
  return <button className="toggle-control" role="switch" aria-checked={checked} onClick={() => onChange(!checked)}><span className={cn("toggle", checked && "toggle-on")}><span /></span><span>{label}</span></button>;
}

function KnowledgeCheck() {
  const questions = [
    { id: "scale", question: "If sequence length doubles, what happens to pairwise interactions?", options: ["They double", "They quadruple", "They stay constant"], answer: 1, explain: "Every token can meet every other token, so N × N becomes (2N) × (2N) = 4N²." },
    { id: "truth", question: "Which computation is the ground truth in this lab?", options: ["The linear approximation", "The BDH sketch", "Exact Softmax Attention"], answer: 2, explain: "We compare the feature-map output against exact Softmax Attention computed from the same Q, K, V." },
    { id: "swap", question: "Why can Linear Attention avoid an N × N matrix?", options: ["It deletes the keys", "It changes the multiplication order around a feature map", "It uses no values"], answer: 1, explain: "The implemented formulation computes φ(K)ᵀV first, producing a d × d state before applying φ(Q)." },
    { id: "same", question: "Is this Linear Attention mathematically identical to Softmax Attention?", options: ["Yes, always", "Only when N is small", "No — it is a specific approximation / formulation"], answer: 2, explain: "A feature-map kernel is not the row-wise softmax kernel. The divergence is a measured property of this toy setup." },
  ];
  const [answers, setAnswers] = useState<Record<string, number>>({});
  return <div className="knowledge-grid">{questions.map((item, index) => { const selected = answers[item.id]; const answered = selected !== undefined; const correct = selected === item.answer; return <div className={cn("knowledge-card", answered && (correct ? "knowledge-correct" : "knowledge-wrong"))} key={item.id}><div className="knowledge-number">0{index + 1}</div><p>{item.question}</p><div className="knowledge-options">{item.options.map((option, optionIndex) => <button key={option} onClick={() => setAnswers((current) => ({ ...current, [item.id]: optionIndex }))} className={cn(answered && optionIndex === item.answer && "knowledge-answer", answered && optionIndex === selected && optionIndex !== item.answer && "knowledge-selected-wrong")}><span>{String.fromCharCode(65 + optionIndex)}</span>{option}</button>)}</div>{answered && <div className="knowledge-explain"><span>{correct ? <Check size={14} /> : <X size={14} />}</span>{item.explain}</div>}</div>; })}</div>;
}

export default function Home() {
  const [sequenceLength, setSequenceLength] = useState(8);
  const [scaleLength, setScaleLength] = useState(8);
  const [featureDimension, setFeatureDimension] = useState(6);
  const [featureMap, setFeatureMap] = useState<FeatureMapName>("elu");
  const [example, setExample] = useState<ExampleName>("sentence");
  const [normalization, setNormalization] = useState(true);
  const [selectedToken, setSelectedToken] = useState(0);
  const [demoMode, setDemoMode] = useState(false);
  const [aboutOpen, setAboutOpen] = useState(false);
  const [divergenceOpen, setDivergenceOpen] = useState(false);
  const [swap, setSwap] = useState(false);
  const [bdhMode, setBdhMode] = useState<"dense" | "sparse">("sparse");
  const [bdhPulse, setBdhPulse] = useState(0);
  const [activeSection, setActiveSection] = useState("why");
  const [copied, setCopied] = useState(false);
  const labRef = useRef<HTMLDivElement>(null);

  const run: AttentionRun = useMemo(() => runAttention(example, sequenceLength, featureDimension, featureMap, normalization), [example, sequenceLength, featureDimension, featureMap, normalization]);
  const points = useMemo(() => complexityPoints(featureDimension, featureMap), [featureDimension, featureMap]);
  const safeSelected = Math.min(selectedToken, run.tokens.length - 1);
  const selectedExactVector = run.exactOutput[safeSelected] ?? [];
  const selectedLinearVector = run.linearOutput[safeSelected] ?? [];
  const differenceMatrix = run.exactOutput.map((row, rowIndex) => row.map((value, columnIndex) => Math.abs(value - (run.linearOutput[rowIndex]?.[columnIndex] ?? 0))));

  useEffect(() => {
    const observers = sectionIds.map((id) => { const node = document.getElementById(id); if (!node) return null; const observer = new IntersectionObserver(([entry]) => { if (entry.isIntersecting) setActiveSection(id); }, { rootMargin: "-18% 0px -62% 0px" }); observer.observe(node); return observer; }).filter(Boolean);
    return () => observers.forEach((observer) => observer?.disconnect());
  }, []);

  useEffect(() => {
    const interval = window.setInterval(() => setBdhPulse((value) => (value + 1) % 8), 720);
    return () => window.clearInterval(interval);
  }, []);

  useEffect(() => {
    const hash = window.location.hash.replace("#experiment=", "");
    if (!hash) return;
    try {
      const config = JSON.parse(decodeURIComponent(hash));
      if (config.sequenceLength) setSequenceLength(Math.min(32, Math.max(4, Number(config.sequenceLength))));
      if (config.featureDimension) setFeatureDimension(Math.min(12, Math.max(2, Number(config.featureDimension))));
      if (config.featureMap) setFeatureMap(config.featureMap);
      if (config.example) setExample(config.example);
    } catch { /* ignore malformed replay state */ }
  }, []);

  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  const startDemo = () => { setDemoMode(true); scrollTo("why"); };
  const reset = () => { setSequenceLength(8); setScaleLength(8); setFeatureDimension(6); setFeatureMap("elu"); setExample("sentence"); setNormalization(true); setSelectedToken(0); setDemoMode(false); setSwap(false); scrollTo("hero"); };
  const replayConfig = { sequenceLength, featureDimension, featureMap, example, normalization };
  const copyReplay = async () => { const value = `${window.location.origin}${window.location.pathname}#experiment=${encodeURIComponent(JSON.stringify(replayConfig))}`; try { await navigator.clipboard.writeText(value); setCopied(true); window.setTimeout(() => setCopied(false), 1800); } catch { setCopied(false); } };

  return <div className="lab-app">
    <header className="topbar"><a className="brand" href="#hero" aria-label="Linear Attention lab home"><span className="brand-mark"><span /><span /><span /></span><span>FRONTIER / LAB</span></a><nav className="topnav" aria-label="Lab sections">{sectionIds.slice(0, 5).map((id, index) => <button key={id} className={activeSection === id ? "nav-active" : ""} onClick={() => scrollTo(id)}><span>0{index + 1}</span>{id === "why" ? "Scale" : id === "softmax" ? "Exact" : id === "linear" ? "Rearrange" : id === "lab" ? "Experiment" : "Limit"}</button>)}</nav><div className="topbar-actions"><button className="icon-button" onClick={() => setAboutOpen(true)} aria-label="About this experiment"><CircleHelp size={17} /></button><button className={cn("demo-button", demoMode && "demo-button-active")} onClick={startDemo}><Play size={14} fill="currentColor" /> {demoMode ? "Demo active" : "Judge demo"}</button></div></header>

    <main>
      <section id="hero" className="hero-section"><div className="hero-orbit orbit-a" /><div className="hero-orbit orbit-b" /><div className="hero-grid" /><div className="container hero-inner"><div className="hero-copy"><div className="kicker"><span className="live-dot" /> LIVE COMPUTATION <span className="kicker-divider" /> No model training required</div><h1>Can attention<br /><em>stay linear?</em></h1><p className="hero-subtitle">Explore the mathematical trade-off between exact Softmax Attention and Linear Attention — by watching the computation happen.</p><div className="hero-cta-row"><button className="primary-button" onClick={() => scrollTo("lab")}>Start the experiment <ArrowRight size={16} /></button><button className="text-button" onClick={() => scrollTo("linear")}>See the mathematics <ArrowDown size={15} /></button></div><div className="hero-footnote"><span className="mono">DATAFORGE 2026 / PATHWAY TRACK</span><span className="hero-footnote-line" /><span className="mono">TOY INPUT · DETERMINISTIC</span></div></div><div className="hero-experiment"><div className="hero-experiment-head"><span>ATTENTION / 001</span><span className="mono">N = 8 · d = 6</span></div><TokenStrip tokens={run.tokens.slice(0, 8)} selected={safeSelected} onSelect={setSelectedToken} /><div className="hero-compare"><div className="hero-method hero-method-exact"><div><span className="method-kicker">STANDARD SOFTMAX ATTENTION</span><strong>O(N²)</strong></div><div className="mini-grid mini-grid-exact">{Array.from({ length: 64 }, (_, index) => <span key={index} style={{ opacity: 0.18 + ((index * 7) % 9) * .07 }} />)}</div><span className="method-caption">pairwise score matrix</span></div><div className="compare-vs">vs</div><div className="hero-method hero-method-linear"><div><span className="method-kicker">LINEAR ATTENTION</span><strong>O(N)</strong></div><div className="mini-flow"><span>φ(K)</span><ArrowRight size={14} /><span className="mini-flow-highlight">KᵀV</span><ArrowRight size={14} /><span>φ(Q)</span></div><span className="method-caption">reordered feature state</span></div></div><div className="hero-equation"><span>softmax(QKᵀ / √d)V</span><span className="equation-arrow">→</span><span className="equation-accent">φ(Q)(φ(K)ᵀV)</span></div></div></div><div className="hero-scroll"><span>scroll to investigate</span><ArrowDown size={14} /></div></section>

      <section id="why" className="section-block"><div className="container"><SectionLabel index="01" eyebrow="THE BOTTLENECK" title="Why attention gets expensive" description="Before we optimize the formula, make the cost visible. Every token can interact with every other token — and those pairwise relationships form a square." /><div className="why-grid"><Panel className="interaction-panel"><div className="panel-head"><div><span className="panel-kicker">INTERACTION FIELD</span><h3>{formatCount(scaleLength * scaleLength)} <small>token-to-token interactions</small></h3></div><Tag tone="warm">N × N</Tag></div><div className="interaction-matrix" style={{ gridTemplateColumns: `repeat(${Math.min(32, scaleLength)}, 1fr)` }} aria-label={`${scaleLength} by ${scaleLength} interaction matrix`}>{Array.from({ length: Math.min(1024, scaleLength * scaleLength) }, (_, index) => <span key={index} className={index % 9 === safeSelected ? "interaction-hot" : ""} style={{ opacity: .28 + ((index * 13) % 7) * .08 }} />)}</div><div className="interaction-caption"><span><i className="legend-dot legend-exact" /> each square = one possible interaction</span><span className="mono">QKᵀ → {scaleLength} × {scaleLength}</span></div></Panel><Panel className="scale-panel"><div className="panel-head"><div><span className="panel-kicker">SEQUENCE LENGTH</span><h3>Watch the square grow.</h3></div><Activity size={18} className="panel-icon" /></div><div className="scale-buttons">{[10, 25, 50, 100, 250, 500].map((length) => <button key={length} onClick={() => setScaleLength(length)} className={scaleLength === length ? "scale-active" : ""}><span>{length}</span><small>{formatCount(length * length)}</small></button>)}</div><div className="scale-readout"><span>Current N</span><strong>{scaleLength}</strong><span className="scale-multiply">×</span><strong>{scaleLength}</strong><span>=</span><strong className="readout-warm">{formatCount(scaleLength * scaleLength)}</strong></div><p className="panel-explanation">Standard self-attention materializes an <strong>N × N</strong> score matrix before it can mix values. That square is the source of the quadratic term.</p><div className="formula-dimension"><span>Q</span><b>{scaleLength} × {featureDimension}</b><span className="formula-arrow">×</span><span>K<sup>T</sup></span><b>{featureDimension} × {scaleLength}</b><span className="formula-arrow">=</span><strong>{scaleLength} × {scaleLength}</strong></div></Panel></div><InfoNote icon={Lightbulb}>What to notice: doubling N does not merely add twice the work — it creates four times as many possible token pairs.</InfoNote></div></section>

      <section id="softmax" className="section-block section-alt"><div className="container"><SectionLabel index="02" eyebrow="THE REFERENCE" title="How Softmax attention works" description="Select a token to follow one real query through the pipeline. This exact computation becomes our ground truth for everything that follows." /><div className="softmax-layout"><Panel className="pipeline-panel"><div className="panel-head"><div><span className="panel-kicker">SELECT A QUERY TOKEN</span><h3>Trace the signal</h3></div><Tag tone="green">computed live</Tag></div><TokenStrip tokens={run.tokens} selected={safeSelected} onSelect={setSelectedToken} /><div className="flow-stack"><FlowStep label="Tokens" formula="x₁ … xₙ" active /><ArrowDown className="flow-arrow" size={16} /><FlowStep label="Project" formula="Q, K, V" active /><ArrowDown className="flow-arrow" size={16} /><FlowStep label="Pairwise scores" formula="QKᵀ / √d" active /><ArrowDown className="flow-arrow" size={16} /><FlowStep label="Normalize" formula="softmax" active /><ArrowDown className="flow-arrow" size={16} /><FlowStep label="Mix values" formula="A · V" active /></div></Panel><Panel className="heatmap-panel"><div className="panel-head"><div><span className="panel-kicker">EXACT ATTENTION WEIGHTS</span><h3>Who does “{run.tokens[safeSelected]}” attend to?</h3></div><Eye size={18} className="panel-icon" /></div><MatrixCard label="A = softmax(QKᵀ / √d)" matrix={run.exactWeights} tokens={run.tokens} selectedRow={safeSelected} /><div className="attention-bars">{run.exactWeights[safeSelected]?.map((weight, index) => <div key={index} className="attention-row"><span>{run.tokens[index]}</span><div><i style={{ width: `${weight * 100}%` }} /></div><b>{(weight * 100).toFixed(1)}%</b></div>)}</div><InfoNote>Read across the highlighted row: softmax turns similarities into a probability distribution, then those weights blend the value vectors.</InfoNote></Panel></div></div></section>

      <section id="linear" className="section-block"><div className="container"><SectionLabel index="03" eyebrow="THE REARRANGEMENT" title="The linear attention trick" description="The expensive part is constructing all pairwise attention scores. A feature map changes the representation so we can regroup the multiplications before an N × N matrix ever exists." /><div className="equation-stage"><Panel className="equation-card equation-card-standard"><div className="equation-card-label"><span>STANDARD</span><span className="mono">pairwise first</span></div><div className="big-equation">softmax(<span>QK<sup>T</sup></span>)V</div><div className="dimension-flow"><span>Q <b>N × d</b></span><ArrowRight size={15} /><span>K<sup>T</sup> <b>d × N</b></span><ArrowRight size={15} /><span className="dim-warning">N × N</span></div><p>Build the attention matrix, normalize each row, then apply it to V.</p></Panel><button className="swap-button" onClick={() => setSwap((value) => !value)} aria-label="Swap multiplication order"><span className={cn(swap && "swap-button-active")}><GitBranch size={19} /></span><small>{swap ? "order swapped" : "swap order"}</small></button><Panel className={cn("equation-card equation-card-linear", swap && "equation-card-linear-active")}><div className="equation-card-label"><span>THIS LAB’S FORMULATION</span><span className="mono">feature map φ</span></div><div className="big-equation">φ(<span>Q</span>){swap ? " · " : " ( "}<span>φ(K)<sup>T</sup>V</span>{swap ? "" : " )"}</div><div className="dimension-flow"><span>φ(K)<sup>T</sup>V <b>d × d</b></span><ArrowRight size={15} /><span>φ(Q) <b>N × d</b></span><ArrowRight size={15} /><span className="dim-good">N × d</span></div><p>Compute a compact feature-state first, then query that state. No explicit N × N score matrix.</p></Panel></div><div className="swap-explanation"><div className="swap-number">{swap ? "02" : "01"}</div><div><strong>{swap ? "The multiplication order has changed." : "Try swapping the order."}</strong><p>{swap ? "Associativity lets us group φ(K)ᵀV first. The intermediate is d × d, so its size depends on the feature dimension — not on every pair of tokens." : "Press the branch button to see the same ingredients regrouped around a feature map. The trick is algebraic, not magical: we avoid materializing the square."}</p></div><Tag tone="blue">specific kernel formulation</Tag></div><details className="technical-details"><summary><Code2 size={15} /> Technical note: what φ means here <ChevronDown size={15} /></summary><p>This explainer uses positive feature maps (ELU + 1, ReLU + ε, or clipped exp) and computes φ(Q)(φ(K)<sup>T</sup>V), with optional denominator normalization. This is one pedagogical kernelized formulation — not a claim that every “Linear Attention” method is identical.</p></details></div></section>

      <section id="lab" ref={labRef} className="section-block lab-section"><div className="container"><div className="lab-title-row"><SectionLabel index="04" eyebrow="LIVE EXPERIMENT LAB" title="Turn the knobs. Watch the math move." description="Every control below feeds the same deterministic Q / K / V inputs into both algorithms. Change a variable and the outputs, error, representations, and cost model all update." /><Tag tone="green"><span className="live-dot" /> live computation</Tag></div><Panel className="controls-panel"><div className="control-item"><label htmlFor="seq-range"><span>Sequence length</span><output>{sequenceLength}</output></label><input id="seq-range" type="range" min="4" max="32" step="1" value={Math.min(sequenceLength, 32)} onChange={(event) => setSequenceLength(Number(event.target.value))} /><div className="range-hint"><span>4</span><span>32 · browser-safe live range</span></div></div><div className="control-item"><label htmlFor="dim-range"><span>Feature dimension</span><output>{featureDimension}</output></label><input id="dim-range" type="range" min="2" max="12" step="1" value={featureDimension} onChange={(event) => setFeatureDimension(Number(event.target.value))} /><div className="range-hint"><span>2</span><span>12</span></div></div><div className="control-item"><label htmlFor="map-select"><span>Feature map</span><output className="output-label">{MAP_COPY[featureMap].label}</output></label><select id="map-select" value={featureMap} onChange={(event) => setFeatureMap(event.target.value as FeatureMapName)}>{Object.entries(MAP_COPY).map(([key, value]) => <option value={key} key={key}>{value.label}</option>)}</select><div className="range-hint"><span>{MAP_COPY[featureMap].description}</span></div></div><div className="control-item"><label htmlFor="example-select"><span>Input example</span><output className="output-label">{EXAMPLE_COPY[example].label}</output></label><select id="example-select" value={example} onChange={(event) => setExample(event.target.value as ExampleName)}>{Object.entries(EXAMPLE_COPY).map(([key, value]) => <option value={key} key={key}>{value.label}</option>)}</select><div className="range-hint"><span>{EXAMPLE_COPY[example].description}</span></div></div><div className="control-item control-toggle"><Toggle checked={normalization} onChange={setNormalization} label="Denominator normalization" /><p>Normalize by φ(q)<sup>T</sup>Σφ(k) to keep output scale comparable.</p></div><button className="reset-button" onClick={reset}><RotateCcw size={15} /> reset</button></Panel><TokenStrip tokens={run.tokens} selected={safeSelected} onSelect={setSelectedToken} /><div className="lab-columns"><Panel className="method-panel exact-panel"><div className="method-panel-head"><div><span className="panel-kicker">REFERENCE / 01</span><h3>Exact Softmax Attention</h3></div><Tag tone="warm">ground truth</Tag></div><div className="method-equation">softmax(<span>QKᵀ / √d</span>)V</div><MatrixCard label="attention weights" matrix={run.exactWeights} tokens={run.tokens} selectedRow={safeSelected} /><VectorBars vector={selectedExactVector} label={`output for “${run.tokens[safeSelected]}”`} tone="warm" /><div className="method-footer"><span><TimerReset size={14} /> {formatCount(run.operations.exact)} modeled ops</span><span>intermediate: {sequenceLength} × {sequenceLength}</span></div></Panel><Panel className="method-panel linear-panel"><div className="method-panel-head"><div><span className="panel-kicker">APPROXIMATION / 02</span><h3>Kernelized Linear Attention</h3></div><Tag tone="blue">feature map φ</Tag></div><div className="method-equation">φ(<span>Q</span>)(φ(<span>K</span>)ᵀV)</div><MatrixCard label={`φ(K) · ${featureMap}`} matrix={run.phiK} tokens={run.tokens} selectedRow={safeSelected} kind="linear" /><VectorBars vector={selectedLinearVector} label={`output for “${run.tokens[safeSelected]}”`} tone="blue" /><div className="method-footer"><span><Zap size={14} /> {formatCount(run.operations.linear)} modeled ops</span><span>state: {featureDimension} × {featureDimension}</span></div></Panel></div><div className="comparison-strip"><div><span className="panel-kicker">SAME INPUTS</span><strong>{sequenceLength} tokens · {featureDimension} dimensions</strong></div><ArrowRight size={18} /><div><span className="panel-kicker">DIVERGENCE</span><strong className={run.mae > .15 ? "text-warm" : "text-blue"}>{run.mae.toFixed(4)} <small>MAE</small></strong></div><ArrowRight size={18} /><div><span className="panel-kicker">TRADE-OFF</span><strong>{(run.operations.exact / run.operations.linear).toFixed(1)}× <small>modeled work</small></strong></div></div><div className="lab-note-row"><InfoNote icon={Gauge}>Operation counts reflect the dominant loops in this implementation. Big-O predicts growth; it is not a promise of device runtime.</InfoNote><button className="outline-button" onClick={() => setDivergenceOpen((value) => !value)}>{divergenceOpen ? <ChevronUp size={15} /> : <ChevronDown size={15} />} how is divergence calculated?</button></div>{divergenceOpen && <Panel className="divergence-drawer"><div><span className="panel-kicker">MAE / MEAN ABSOLUTE ERROR</span><h3>One number, shown honestly.</h3><p>For every output coordinate, subtract the linear value from the exact value, take the absolute value, then average across all tokens and dimensions.</p></div><div className="divergence-formula"><span>MAE</span><b>=</b><span className="fraction"><i>Σ | exact − linear |</i><em>N × d</em></span><b>=</b><strong>{run.mae.toFixed(4)}</strong></div></Panel>}</div></section>

      <section id="failure" className="section-block section-alt"><div className="container"><SectionLabel index="05" eyebrow="THE LIMITATION" title="Now make it fail on purpose" description="A useful explainer should show not only the speed story, but where a specific approximation can diverge. Here, a query must retrieve one needle from a field of distractors." /><div className="failure-grid"><Panel className="retrieval-panel"><div className="panel-head"><div><span className="panel-kicker">RETRIEVAL PROBE</span><h3>Find the needle</h3></div><Tag tone="warm"><TriangleAlert size={13} /> limitation</Tag></div><div className="retrieval-sequence">{run.tokens.map((token, index) => <button key={`${token}-${index}`} className={cn("retrieval-token", index === run.topExact && "retrieval-exact", index === run.topLinear && "retrieval-linear", index === run.tokens.length - 2 && "retrieval-query")} onClick={() => setSelectedToken(index)}><small>{String(index + 1).padStart(2, "0")}</small><span>{token}</span>{index === run.topExact && <b>exact</b>}{index === run.topLinear && <b>φ</b>}</button>)}</div><div className="retrieval-key"><span><i className="key-dot key-exact" /> Softmax pick</span><span><i className="key-dot key-linear" /> Linear score peak</span><span><i className="key-dot key-query" /> query</span></div><div className="retrieval-readout"><div><span>WHAT WE EXPECT</span><strong>needle:42</strong></div><div><span>WHAT SOFTMAX DOES</span><strong className="text-warm">{run.tokens[run.topExact]}</strong></div><div><span>WHAT LINEAR DOES</span><strong className="text-blue">{run.tokens[run.topLinear]}</strong></div></div></Panel><Panel className="failure-explain"><div className="failure-icon"><Network size={21} /></div><span className="panel-kicker">WHY DID IT DIVERGE?</span><h3>Different similarity geometry.</h3><p>Softmax uses a row-wise exponential over pairwise dot products. This lab’s positive feature map instead factors a kernel and aggregates a compact state. The two geometries can rank distractors differently — especially when the query is designed to be sharp.</p><div className="failure-steps"><div><b>01</b><span>Exact sees a full score row before normalizing.</span></div><div><b>02</b><span>Linear sees feature interactions through φ(K)<sup>T</sup>V.</span></div><div><b>03</b><span>The approximation is formulation- and task-dependent.</span></div></div><InfoNote icon={TriangleAlert} tone="warm">This is a pedagogical counterexample, not a universal claim that Linear Attention always fails on long sequences.</InfoNote><button className="text-button" onClick={() => { setExample(example === "retrieval" ? "sentence" : "retrieval"); scrollTo("lab"); }}>Change the experiment <ArrowRight size={15} /></button></Panel></div></div></section>

      <section id="bdh" className="section-block bdh-section"><div className="container"><SectionLabel index="06" eyebrow="WHY PATHWAY CARES" title="The connection is architectural, not cosmetic." description="The rearrangement you just saw points toward an important design question: where should information live when context grows? BDH explores a different answer — sparse activity, recurrent state, and synaptic / fast-weight-like memory." /><div className="bdh-intro-grid"><Panel className="bdh-story"><div className="bdh-badge"><BrainCircuit size={19} /><span>PATHWAY / BDH-CQ</span></div><h3>BDH is not simply Linear Attention with a different name.</h3><p>BDH — Dragon Hatchling — is described by Pathway as a post-Transformer, brain-inspired architecture. It moves beyond a conventional layer that materializes global attention by combining sparse positive activations, local interactions, recurrent latent state, and memory-like synaptic updates.</p><div className="bdh-pill-row"><Tag tone="blue">sparse positive activity</Tag><Tag tone="blue">recurrent state</Tag><Tag tone="blue">synaptic memory</Tag></div><a href="https://pathway.com/research/bdh-explainer" target="_blank" rel="noreferrer" className="source-link">Read Pathway’s BDH explainer <ExternalLink size={13} /></a></Panel><Panel className="bdh-network-panel"><div className="panel-head"><div><span className="panel-kicker">CONCEPTUAL / PEDAGOGICAL VISUALIZATION</span><h3>Where information flows</h3></div><Sparkles size={18} className="panel-icon" /></div><div className="bdh-toggle-row"><button className={bdhMode === "dense" ? "bdh-mode-active" : ""} onClick={() => setBdhMode("dense")}><span className="mode-swatch mode-dense" /> dense / conventional interaction</button><button className={bdhMode === "sparse" ? "bdh-mode-active" : ""} onClick={() => setBdhMode("sparse")}><span className="mode-swatch mode-sparse" /> sparse / local + stateful</button></div><BdhNetwork mode={bdhMode} pulse={bdhPulse} /><div className="network-caption"><span>{bdhMode === "dense" ? "many possible edges" : "selected active pathways"}</span><span className="mono">t = {bdhPulse + 1}</span></div><InfoNote icon={Info}>Conceptual only: this graphic is not an official BDH implementation or an equation-level simulator.</InfoNote></Panel></div><div className="comparison-table-wrap"><div className="table-title"><span className="panel-kicker">THREE WAYS TO ORGANIZE CONTEXT</span><Tag tone="muted">source-backed distinctions</Tag></div><div className="comparison-table" role="table"><div className="comparison-row comparison-head" role="row"><span>DIMENSION</span><span>TRANSFORMER</span><span>LINEAR ATTENTION</span><span>BDH</span></div>{[["Attention mechanism", "dense softmax over token pairs", "kernel / feature-map formulation", "local interactions + learned dynamics"], ["Memory / state", "activations in the context window", "compressed φ(K)ᵀV state", "recurrent latent state + synaptic memory"], ["Scaling behavior", "explicit N × N scores", "linear in N for fixed d", "designed around sparse, recurrent computation"], ["Sparsity", "typically dense activations", "depends on chosen feature map", "sparse, positive activations are a design principle"], ["Where info lives", "tokens and layer activations", "feature-state accumulator", "neurons, recurrent state, and synaptic connections"]].map((row) => <div className="comparison-row" key={row[0]} role="row">{row.map((cell, index) => <span key={cell} className={index === 0 ? "row-label" : ""}>{cell}</span>)}</div>)}</div></div><div className="bdh-bottom-note"><InfoNote icon={Link2}>The resemblance: both can replace an explicit all-pairs attention matrix with a state-like computation. The difference: BDH is a broader architecture with its own mechanisms for sparse activation, recurrence, adaptation, and memory.</InfoNote><div className="source-inline"><a href="https://github.com/pathwaycom/bdh" target="_blank" rel="noreferrer">BDH code / paper resources <ExternalLink size={12} /></a><a href="https://pathway.com/research/introducing-bdh-cq" target="_blank" rel="noreferrer">BDH-CQ overview <ExternalLink size={12} /></a></div></div></div></section>

      <section id="takeaway" className="section-block takeaway-section"><div className="container"><SectionLabel index="07" eyebrow="SYNTHESIS" title="What did we actually learn?" description="The efficiency story is real. So is the trade-off. Keep both in view." /><div className="tradeoff-wrap"><div className="tradeoff-axis"><div className="axis-line" /><span className="axis-top">COMPUTATIONAL<br />EFFICIENCY</span><span className="axis-bottom">APPROXIMATION<br />TRADE-OFF</span><div className="tradeoff-node"><span>LINEAR<br />ATTENTION</span></div></div><div className="tradeoff-copy"><p>Reorganizing a computation can change its scaling without preserving every property of the original operation.</p><div className="tradeoff-points"><div><Gauge size={16} /><span><b>Efficiency</b> — avoid explicit pairwise scores.</span></div><div><TriangleAlert size={16} /><span><b>Fidelity</b> — measure the approximation against ground truth.</span></div><div><BrainCircuit size={16} /><span><b>Architecture</b> — BDH explores a different state-and-memory design space.</span></div></div></div></div><div className="learned-list"><div className="learned-list-head"><span className="panel-kicker">THE TAKEAWAY, REVEALED</span><span className="mono">7 observations</span></div>{["Softmax Attention creates pairwise interactions.", "Pairwise interaction creates O(N²) scaling.", "Kernel / feature-map formulations can rearrange computation.", "Linear Attention reduces scaling cost for fixed feature dimension.", "The approximation can differ from exact Softmax behavior.", "The limitation depends on the formulation and task.", "BDH explores efficient reasoning and memory through a different architectural approach."].map((item, index) => <div key={item} className="learned-row"><span>{String(index + 1).padStart(2, "0")}</span><p>{item}</p><Check size={15} /></div>)}</div><div className="replay-panel"><div><span className="panel-kicker">EXPERIMENT REPLAY</span><h3>Make your result reproducible.</h3><p>Copy this configuration to reopen the same toy experiment later or share it with a teammate.</p></div><div className="replay-config"><span>N = {sequenceLength}</span><span>d = {featureDimension}</span><span>{MAP_COPY[featureMap].label.split(" · ")[0]}</span><span>{EXAMPLE_COPY[example].label}</span></div><button className="primary-button" onClick={copyReplay}>{copied ? <Check size={16} /> : <Clipboard size={16} />} {copied ? "Copied replay link" : "Copy replay link"}</button></div><div className="knowledge-section"><div className="knowledge-heading"><div><span className="panel-kicker">MINI KNOWLEDGE CHECK</span><h3>Can you predict the next move?</h3></div><Lightbulb size={18} /></div><KnowledgeCheck /></div><div className="sources-section"><div><span className="panel-kicker">SOURCES & TECHNICAL NOTES</span><h3>What is live, what is conceptual?</h3></div><div className="source-columns"><div><Tag tone="green">LIVE COMPUTATION</Tag><p>Q, K, V generation, softmax weights, feature maps, outputs, MAE / L2 divergence, and modeled operation counts run in the browser.</p></div><div><Tag tone="blue">CONCEPTUAL</Tag><p>The BDH network graphic is pedagogical. It communicates mechanisms described by Pathway but is not an official BDH implementation.</p></div><div><Tag tone="muted">PRIMARY RESEARCH</Tag><p><a href="https://arxiv.org/abs/2006.16236" target="_blank" rel="noreferrer">Katharopoulos et al., 2020</a> for linear attention; <a href="https://arxiv.org/html/1706.03762v7" target="_blank" rel="noreferrer">Vaswani et al., 2017</a> for the Softmax Attention reference; <a href="https://pathway.com/research/bdh-explainer" target="_blank" rel="noreferrer">Pathway’s BDH explainer</a> and <a href="https://github.com/pathwaycom/bdh" target="_blank" rel="noreferrer">BDH repository</a> for the architecture connection.</p></div></div><p className="methodology-footer"><Info size={14} /> This is a pedagogical implementation, not a production Transformer or official BDH implementation. It does not claim benchmark performance.</p></div></div></section>
    </main>

    <footer className="site-footer"><div className="container"><div className="footer-brand"><span className="brand-mark"><span /><span /><span /></span><span>FRONTIER / LAB</span></div><span className="mono">LINEAR ATTENTION — SEE THE APPROXIMATION</span><span className="mono">DATAFORGE 2026 · PATHWAY TRACK</span></div></footer>

    {aboutOpen && <div className="drawer-backdrop" onClick={() => setAboutOpen(false)}><aside className="about-drawer" onClick={(event) => event.stopPropagation()}><div className="drawer-head"><div><span className="panel-kicker">METHOD / 001</span><h3>About this experiment</h3></div><button className="icon-button" onClick={() => setAboutOpen(false)} aria-label="Close"><X size={17} /></button></div><p>This lab uses small deterministic toy inputs so every interaction is fast and inspectable. It is designed to teach the computational idea — not to reproduce a production Transformer benchmark.</p><div className="drawer-list"><div><Tag tone="green">LIVE</Tag><span>Exact and linear outputs are recalculated client-side from the same Q / K / V inputs.</span></div><div><Tag tone="blue">MODELED</Tag><span>Operation counts follow the dominant loops in this implementation, not noisy wall-clock measurements.</span></div><div><Tag tone="muted">ILLUSTRATION</Tag><span>The BDH network and comparison are sourced explanations plus a conceptual visualization.</span></div></div><div className="drawer-divider" /><h4>Assumptions</h4><ul><li>Single-head toy attention; no learned weights or training.</li><li>Linear Attention is the positive feature-map formulation shown in the equation viewer.</li><li>BDH claims are qualified to distinguish documented architecture from this demo.</li></ul><a className="source-link" href="https://pathway.com/research/bdh-explainer" target="_blank" rel="noreferrer">Open source notes <ExternalLink size={13} /></a></aside></div>}
  </div>;
}
