"use client";

import { useEffect, useMemo, useState } from "react";

function mulberry32(seed: number) {
  let t = seed >>> 0;
  return function () {
    t += 0x6d2b79f5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function randn(rng: () => number) {
  // Box–Muller
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

type Point = { x: number; y: number };

type ExplainLinearResponse = {
  mse: number;
  grad_w: number;
  grad_b: number;
  errors: number[];
};

type SurfaceLinearResponse = {
  w_values: number[];
  b_values: number[];
  mse_grid: number[][];
};

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

type Viewport = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

function worldToSvg(
  x: number,
  y: number,
  vp: Viewport,
  width: number,
  height: number,
) {
  const px = ((x - vp.xMin) / (vp.xMax - vp.xMin)) * width;
  const py = height - ((y - vp.yMin) / (vp.yMax - vp.yMin)) * height;
  return { x: px, y: py };
}

export default function Home() {
  const [w, setW] = useState<number>(1);
  const [b, setB] = useState<number>(0);
  const [points, setPoints] = useState<Point[]>([]);
  const [mse, setMse] = useState<number>(0);
  const [isLoadingPoints, setIsLoadingPoints] = useState<boolean>(true);
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [epoch, setEpoch] = useState<number>(0);
  const [iteration, setIteration] = useState<number>(0);
  const [lossHistory, setLossHistory] = useState<number[]>([]);
  const [wHistory, setWHistory] = useState<number[]>([]);
  const [bHistory, setBHistory] = useState<number[]>([]);
  const [errors, setErrors] = useState<number[]>([]);
  const [gradW, setGradW] = useState<number>(0);
  const [gradB, setGradB] = useState<number>(0);
  const [lastStep, setLastStep] = useState<
    | {
        wPrev: number;
        bPrev: number;
        wNext: number;
        bNext: number;
        gradW: number;
        gradB: number;
        lr: number;
        loss: number;
      }
    | null
  >(null);

  const [trainMode, setTrainMode] = useState<"full" | "mini" | "sgd">("full");
  const [miniBatchSize, setMiniBatchSize] = useState<number>(8);

  const [surface, setSurface] = useState<SurfaceLinearResponse | null>(null);
  const [isLoadingSurface, setIsLoadingSurface] = useState<boolean>(false);

  const width = 1100;
  const height = 620;
  // For w,b in [-5,5] and x in [-5,5], y roughly stays within [-30,30]
  const vp: Viewport = { xMin: -5, xMax: 5, yMin: -30, yMax: 30 };

  const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://localhost:8000";

  useEffect(() => {
    let isMounted = true;
    const controller = new AbortController();

    async function loadPoints() {
      setIsLoadingPoints(true);
      try {
        const res = await fetch(`${API_BASE}/dataset/linear`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ n: 40, seed: 42 }),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`dataset error: ${res.status}`);
        const data = (await res.json()) as { points: Point[] };
        if (isMounted) setPoints(data.points);
      } catch {
        if (isMounted) setPoints([]);
      } finally {
        if (isMounted) setIsLoadingPoints(false);
      }
    }

    void loadPoints();
    return () => {
      isMounted = false;
      controller.abort();
    };
  }, [API_BASE]);

  useEffect(() => {
    if (points.length === 0) {
      setSurface(null);
      return;
    }

    const controller = new AbortController();
    async function loadSurface() {
      setIsLoadingSurface(true);
      try {
        const res = await fetch(`${API_BASE}/surface/linear/mse`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            points,
            w_min: -5,
            w_max: 5,
            b_min: -5,
            b_max: 5,
            w_steps: 25,
            b_steps: 25,
          }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`surface error: ${res.status}`);
        const data = (await res.json()) as SurfaceLinearResponse;
        setSurface(data);
      } catch {
        setSurface(null);
      } finally {
        setIsLoadingSurface(false);
      }
    }

    void loadSurface();
    return () => controller.abort();
  }, [API_BASE, points]);

  useEffect(() => {
    if (points.length === 0) {
      setMse(0);
      setErrors([]);
      setGradW(0);
      setGradB(0);
      return;
    }

    if (isTraining) return;

    const controller = new AbortController();
    async function loadExplain() {
      try {
        const res = await fetch(`${API_BASE}/explain/linear`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ w, b, points }),
          signal: controller.signal,
        });

        if (!res.ok) throw new Error(`explain error: ${res.status}`);
        const data = (await res.json()) as ExplainLinearResponse;
        setMse(data.mse);
        setErrors(data.errors);
        setGradW(data.grad_w);
        setGradB(data.grad_b);
      } catch {
        setMse(0);
        setErrors([]);
        setGradW(0);
        setGradB(0);
      }
    }

    void loadExplain();
    return () => controller.abort();
  }, [API_BASE, points, w, b, isTraining]);

  const learningRate = 0.03;

  async function runTrainStep(wPrev: number, bPrev: number, batchPoints: Point[]) {
    const res = await fetch(`${API_BASE}/train/linear/step`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        w: wPrev,
        b: bPrev,
        learning_rate: learningRate,
        points: batchPoints,
      }),
    });

    if (!res.ok) throw new Error(`train error: ${res.status}`);
    const data = (await res.json()) as {
      w: number;
      b: number;
      loss: number;
      grad_w: number;
      grad_b: number;
      errors: number[];
    };

    const wNext = data.w;
    const bNext = data.b;

    setW(wNext);
    setB(bNext);
    setMse(data.loss);
    setErrors(data.errors);
    setGradW(data.grad_w);
    setGradB(data.grad_b);
    setWHistory((prev) => (prev.length === 0 ? [wPrev, wNext] : [...prev, wNext]));
    setBHistory((prev) => (prev.length === 0 ? [bPrev, bNext] : [...prev, bNext]));

    setLastStep({
      wPrev,
      bPrev,
      wNext,
      bNext,
      gradW: data.grad_w,
      gradB: data.grad_b,
      lr: learningRate,
      loss: data.loss,
    });

    setIteration((prev) => prev + 1);

    return { wNext, bNext };
  }

  function getBatchSize(n: number) {
    if (trainMode === "full") return n;
    if (trainMode === "sgd") return 1;
    return Math.max(1, Math.min(n, Math.floor(miniBatchSize || 1)));
  }

  function makeShuffledIndices(n: number, seed: number) {
    const rng = mulberry32(seed);
    const idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  async function trainOneIteration() {
    if (isTraining || isLoadingPoints || points.length === 0) return;
    setIsTraining(true);
    try {
      const n = points.length;
      const bs = getBatchSize(n);
      // Deterministic mini-batch selection based on current (epoch, iteration)
      const shuffled = makeShuffledIndices(n, 1234 + epoch);
      const iterInEpoch = iteration % Math.ceil(n / bs);
      const start = iterInEpoch * bs;
      const batchIdx = shuffled.slice(start, start + bs);
      const batchPoints = batchIdx.map((i) => points[i]);

      await runTrainStep(w, b, batchPoints);
    } finally {
      setIsTraining(false);
    }
  }

  async function evaluateEpochLoss(wNow: number, bNow: number) {
    const res = await fetch(`${API_BASE}/explain/linear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ w: wNow, b: bNow, points }),
    });
    if (!res.ok) throw new Error(`explain error: ${res.status}`);
    const data = (await res.json()) as ExplainLinearResponse;
    setMse(data.mse);
    setErrors(data.errors);
    setGradW(data.grad_w);
    setGradB(data.grad_b);
    setLossHistory((prev) => [...prev, data.mse]);
  }

  async function trainOneEpoch() {
    if (isTraining || isLoadingPoints || points.length === 0) return;
    setIsTraining(true);
    try {
      const n = points.length;
      const bs = getBatchSize(n);
      const itersPerEpoch = Math.ceil(n / bs);
      const shuffled = makeShuffledIndices(n, 1234 + epoch);

      let currentW = w;
      let currentB = b;
      for (let i = 0; i < itersPerEpoch; i++) {
        const start = i * bs;
        const batchIdx = shuffled.slice(start, start + bs);
        const batchPoints = batchIdx.map((j) => points[j]);
        const next = await runTrainStep(currentW, currentB, batchPoints);
        currentW = next.wNext;
        currentB = next.bNext;
      }

      setEpoch((prev) => prev + 1);
      await evaluateEpochLoss(currentW, currentB);
    } finally {
      setIsTraining(false);
    }
  }

  async function trainGradientDescent() {
    if (isTraining || points.length === 0) return;
    setIsTraining(true);
    setLossHistory([]);
    setWHistory([w]);
    setBHistory([b]);
    setEpoch(0);
    setIteration(0);
    setLastStep(null);

    const epochs = 30;
    let currentW = w;
    let currentB = b;

    try {
      for (let e = 0; e < epochs; e++) {
        const n = points.length;
        const bs = getBatchSize(n);
        const itersPerEpoch = Math.ceil(n / bs);
        const shuffled = makeShuffledIndices(n, 1234 + e);

        for (let i = 0; i < itersPerEpoch; i++) {
          const start = i * bs;
          const batchIdx = shuffled.slice(start, start + bs);
          const batchPoints = batchIdx.map((j) => points[j]);
          const next = await runTrainStep(currentW, currentB, batchPoints);
          currentW = next.wNext;
          currentB = next.bNext;
          await sleep(50);
        }

        setEpoch((prev) => prev + 1);
        await evaluateEpochLoss(currentW, currentB);
      }
    } finally {
      setIsTraining(false);
    }
  }

  function resetTrainingHistory() {
    setEpoch(0);
    setIteration(0);
    setLossHistory([]);
    setWHistory([]);
    setBHistory([]);
    setLastStep(null);
  }

  const line = useMemo(() => {
    const x1 = vp.xMin;
    const y1 = w * x1 + b;
    const x2 = vp.xMax;
    const y2 = w * x2 + b;

    const p1 = worldToSvg(x1, y1, vp, width, height);
    const p2 = worldToSvg(x2, y2, vp, width, height);
    return { p1, p2 };
  }, [w, b]);

  const gridXs = useMemo(() => {
    const xs: number[] = [];
    for (let x = Math.ceil(vp.xMin); x <= Math.floor(vp.xMax); x++) xs.push(x);
    return xs;
  }, [vp.xMin, vp.xMax]);

  const gridYs = useMemo(() => {
    const ys: number[] = [];
    for (let y = Math.ceil(vp.yMin / 5) * 5; y <= Math.floor(vp.yMax / 5) * 5; y += 5)
      ys.push(y);
    return ys;
  }, [vp.yMin, vp.yMax]);

  const axis = useMemo(() => {
    const p0 = worldToSvg(0, 0, vp, width, height);
    return { x0: p0.x, y0: p0.y };
  }, [vp, width, height]);

  const errorSegments = useMemo(() => {
    if (errors.length !== points.length) return [] as Array<{
      x: number;
      yTrue: number;
      yHat: number;
      err: number;
    }>;

    return points.map((p, idx) => {
      const err = errors[idx] ?? 0;
      return { x: p.x, yTrue: p.y, yHat: p.y + err, err };
    });
  }, [errors, points]);

  const gradArrow = useMemo(() => {
    // We move along -grad to decrease loss
    const dx = -gradW;
    const dy = -gradB;
    const mag = Math.hypot(dx, dy);
    if (!Number.isFinite(mag) || mag <= 1e-12) {
      return { x2: 0, y2: 0, mag: 0 };
    }
    return { x2: dx / mag, y2: dy / mag, mag };
  }, [gradW, gradB]);

  const surfaceWireframe = useMemo(() => {
    if (!surface || surface.w_values.length === 0 || surface.b_values.length === 0) return null;

    const wVals = surface.w_values;
    const bVals = surface.b_values;
    const Z = surface.mse_grid;
    const wN = wVals.length;
    const bN = bVals.length;

    let zMin = Infinity;
    let zMax = -Infinity;
    for (let i = 0; i < wN; i++) {
      for (let j = 0; j < bN; j++) {
        const v = Z[i]?.[j];
        if (typeof v !== "number" || !Number.isFinite(v)) continue;
        zMin = Math.min(zMin, v);
        zMax = Math.max(zMax, v);
      }
    }
    const span = Math.max(1e-9, zMax - zMin);

    const sw = 900;
    const sh = 420;
    const ox = sw / 2;
    const oy = 260;
    const sx = 38; // iso scale
    const sy = 18;
    const sz = 160;

    const w0 = wVals[0];
    const w1 = wVals[wN - 1];
    const b0 = bVals[0];
    const b1 = bVals[bN - 1];
    const wSpan = Math.max(1e-9, w1 - w0);
    const bSpan = Math.max(1e-9, b1 - b0);

    const project = (wV: number, bV: number, zV: number) => {
      const wx = ((wV - w0) / wSpan - 0.5) * 2;
      const bx = ((bV - b0) / bSpan - 0.5) * 2;
      const zn = (zV - zMin) / span;
      const x = ox + (wx - bx) * sx;
      const y = oy + (wx + bx) * sy - zn * sz;
      return { x, y, zn };
    };

    const wLines = wVals.map((wV, i) => {
      const pts = bVals
        .map((bV, j) => {
          const p = project(wV, bV, Z[i][j]);
          return `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
        })
        .join(" ");
      return { key: `w-${i}`, pts };
    });

    const bLines = bVals.map((bV, j) => {
      const pts = wVals
        .map((wV, i) => {
          const p = project(wV, bV, Z[i][j]);
          return `${p.x.toFixed(2)},${p.y.toFixed(2)}`;
        })
        .join(" ");
      return { key: `b-${j}`, pts };
    });

    // Gradient arrow projected on base plane at current (w,b)
    const baseLoss = zMin;
    const p0 = project(w, b, baseLoss);
    const arrowScale = 0.35;
    const p1 = project(w + gradArrow.x2 * arrowScale, b + gradArrow.y2 * arrowScale, baseLoss);

    return { sw, sh, wLines, bLines, p0, p1 };
  }, [surface, w, b, gradArrow]);

  const lossStats = useMemo(() => {
    if (lossHistory.length === 0) return { current: null as number | null, best: null as number | null };
    const best = Math.min(...lossHistory);
    const current = lossHistory[lossHistory.length - 1];
    return { current, best };
  }, [lossHistory]);

  return (
    <div className="min-h-screen bg-zinc-50 font-sans text-zinc-900 dark:bg-black dark:text-zinc-50">
      <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-8 py-10">
        <header className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold tracking-tight">線性模型視覺化</h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            調整參數並觀察直線：y = wx + b
          </p>
          <div className="mt-2 rounded-2xl border border-zinc-200 bg-white p-4 text-sm text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-300">
            <div className="grid gap-1">
              <div>
                你會在主圖看到每個點的誤差：<span className="font-mono">e = ŷ - y</span>（顏色區分正負），以及在左側看到梯度
                <span className="font-mono">(∂L/∂w, ∂L/∂b)</span>。
              </div>
              <div>
                梯度下降更新規則是 <span className="font-mono">w ← w - lr·∂L/∂w</span>、<span className="font-mono">b ← b - lr·∂L/∂b</span>（這就是你想看的 backprop 在做的事）。
              </div>
            </div>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[360px_1fr]">
          <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
            <div className="mb-4 rounded-xl bg-zinc-50 p-3 font-mono text-sm text-zinc-900 dark:bg-zinc-900/40 dark:text-zinc-50">
              y = <span className="font-semibold">{w.toFixed(2)}</span>x +{" "}
              <span className="font-semibold">{b.toFixed(2)}</span>
            </div>

            <div className="mb-4 flex items-baseline justify-between rounded-xl border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-950">
              <div className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                Loss (MSE)
              </div>
              <div className="font-mono text-sm">
                {isLoadingPoints ? "loading..." : mse.toFixed(4)}
              </div>
            </div>

            <div className="mb-4 flex flex-col gap-2 rounded-xl border border-zinc-200 bg-white p-3 dark:border-zinc-800 dark:bg-zinc-950">
              <div className="flex items-baseline justify-between">
                <div className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Training
                </div>
                <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                  start from current w,b
                </div>
              </div>

              <div className="mt-1 rounded-lg border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
                <div className="flex items-center justify-between gap-3">
                  <label className="flex items-center gap-2">
                    <span className="text-[11px] text-zinc-600 dark:text-zinc-400">Mode</span>
                    <select
                      value={trainMode}
                      onChange={(e) => {
                        setTrainMode(e.target.value as "full" | "mini" | "sgd");
                        resetTrainingHistory();
                      }}
                      className="h-9 rounded-lg border border-zinc-200 bg-white px-2 text-xs dark:border-zinc-800 dark:bg-zinc-950"
                      disabled={isTraining}
                    >
                      <option value="full">Full Batch</option>
                      <option value="mini">Mini-batch</option>
                      <option value="sgd">SGD (batch=1)</option>
                    </select>
                  </label>

                  <label className="flex items-center gap-2">
                    <span className="text-[11px] text-zinc-600 dark:text-zinc-400">Batch</span>
                    <input
                      type="number"
                      min={1}
                      max={points.length || 1}
                      value={miniBatchSize}
                      onChange={(e) => {
                        setMiniBatchSize(Math.max(1, Math.floor(Number(e.target.value || 1))));
                        resetTrainingHistory();
                      }}
                      className="h-9 w-20 rounded-lg border border-zinc-200 bg-white px-2 text-xs dark:border-zinc-800 dark:bg-zinc-950"
                      disabled={isTraining || trainMode !== "mini"}
                    />
                  </label>
                </div>
                <div className="mt-2 grid gap-1 text-[11px] text-zinc-600 dark:text-zinc-400">
                  <div>
                    <span className="font-medium">Iteration</span> = 一次參數更新（一次 mini-batch/SGD step）。
                  </div>
                  <div>
                    <span className="font-medium">Epoch</span> = 把資料「完整看過一輪」：Full batch 是 1 iteration；mini-batch/SGD 會有多次 iteration。
                  </div>
                </div>
              </div>

              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={trainGradientDescent}
                    disabled={isTraining || isLoadingPoints || points.length === 0}
                    className="h-10 rounded-full bg-foreground px-4 text-sm font-medium text-background disabled:cursor-not-allowed disabled:opacity-60"
                  >
                    {isTraining ? "Training..." : "Auto Train"}
                  </button>
                  <button
                    type="button"
                    onClick={trainOneIteration}
                    disabled={isTraining || isLoadingPoints || points.length === 0}
                    className="h-10 rounded-full border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-900 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50"
                  >
                    Step 1 Iteration
                  </button>
                  <button
                    type="button"
                    onClick={trainOneEpoch}
                    disabled={isTraining || isLoadingPoints || points.length === 0}
                    className="h-10 rounded-full border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-900 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50"
                  >
                    Step 1 Epoch
                  </button>
                </div>
                <div className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
                  epoch: {epoch} · iter: {iteration}
                </div>
              </div>
              <div className="font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
                lr: {learningRate}
              </div>
              <div className="flex items-baseline justify-between font-mono text-xs text-zinc-600 dark:text-zinc-400">
                <div>
                  current: {lossStats.current === null ? "-" : lossStats.current.toFixed(4)}
                </div>
                <div>best: {lossStats.best === null ? "-" : lossStats.best.toFixed(4)}</div>
              </div>

              <div className="mt-2 rounded-lg border border-zinc-200 bg-zinc-50 p-3 dark:border-zinc-800 dark:bg-zinc-900/40">
                <div className="mb-2 flex items-baseline justify-between">
                  <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
                    Backprop (Gradients)
                  </div>
                  <div className="font-mono text-[11px] text-zinc-600 dark:text-zinc-400">
                    update = -(lr · grad)
                  </div>
                </div>
                <div className="grid gap-2 font-mono text-xs text-zinc-700 dark:text-zinc-300">
                  <div className="flex justify-between">
                    <span>∂L/∂w</span>
                    <span>{gradW.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>∂L/∂b</span>
                    <span>{gradB.toFixed(4)}</span>
                  </div>
                </div>
                <div className="mt-3 flex items-center justify-between gap-3">
                  <div className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
                    w,b step: {lastStep ? `${lastStep.wPrev.toFixed(3)}→${lastStep.wNext.toFixed(3)}, ${lastStep.bPrev.toFixed(3)}→${lastStep.bNext.toFixed(3)}` : "-"}
                  </div>
                  <svg viewBox="0 0 120 70" className="h-10 w-20" role="img" aria-label="gradient direction">
                    <defs>
                      <marker
                        id="arrow"
                        viewBox="0 0 10 10"
                        refX="8"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto-start-reverse"
                      >
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
                      </marker>
                    </defs>
                    <rect x="1" y="1" width="118" height="68" rx="10" ry="10" fill="transparent" stroke="currentColor" className="text-zinc-200 dark:text-zinc-800" />
                    <line x1="60" y1="35" x2="60" y2="10" stroke="currentColor" className="text-zinc-200 dark:text-zinc-800" strokeWidth="2" />
                    <line x1="60" y1="35" x2="110" y2="35" stroke="currentColor" className="text-zinc-200 dark:text-zinc-800" strokeWidth="2" />
                    <text x="62" y="14" fontSize="10" fill="currentColor" className="text-zinc-500 dark:text-zinc-400">-∂b</text>
                    <text x="90" y="47" fontSize="10" fill="currentColor" className="text-zinc-500 dark:text-zinc-400">-∂w</text>
                    <line
                      x1={60}
                      y1={35}
                      x2={60 + gradArrow.x2 * 28}
                      y2={35 - gradArrow.y2 * 28}
                      stroke="currentColor"
                      className="text-emerald-600 dark:text-emerald-400"
                      strokeWidth="3"
                      markerEnd="url(#arrow)"
                      strokeLinecap="round"
                    />
                  </svg>
                </div>
              </div>
            </div>

            <div className="flex flex-col gap-5">
              <label className="flex flex-col gap-2">
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-medium">w（斜率）</span>
                  <span className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
                    {w.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={-5}
                  max={5}
                  step={0.01}
                  value={w}
                  onChange={(e) => {
                    setW(parseFloat(e.target.value));
                    resetTrainingHistory();
                  }}
                  className="w-full"
                  disabled={isTraining}
                />
              </label>

              <label className="flex flex-col gap-2">
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-medium">b（截距）</span>
                  <span className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
                    {b.toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={-5}
                  max={5}
                  step={0.01}
                  value={b}
                  onChange={(e) => {
                    setB(parseFloat(e.target.value));
                    resetTrainingHistory();
                  }}
                  className="w-full"
                  disabled={isTraining}
                />
              </label>

              <div className="text-xs text-zinc-600 dark:text-zinc-400">
                調整滑桿會重置訓練軌跡（因為起點改了）。
              </div>
            </div>
          </div>

          <div className="grid gap-6">
            <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
              <div className="mb-3 flex items-baseline justify-between">
                <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">Loss Surface (w,b) — 3D view</h2>
                <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                  {isLoadingSurface ? "loading..." : ""}
                </div>
              </div>
              <div className="mb-2 text-sm text-zinc-600 dark:text-zinc-400">
                這張「山」是 <span className="font-mono">L(w,b)</span>：高度越高代表 loss 越大。綠色箭頭是 <span className="font-mono">-∇L</span>（微分告訴你往哪裡走會下降）。
              </div>
              <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
                {surfaceWireframe ? (
                  <svg
                    viewBox={`0 0 ${surfaceWireframe.sw} ${surfaceWireframe.sh}`}
                    className="h-auto w-full"
                    role="img"
                    aria-label="loss surface"
                  >
                    <rect x={0} y={0} width={surfaceWireframe.sw} height={surfaceWireframe.sh} fill="transparent" />
                    {surfaceWireframe.wLines.map((l) => (
                      <polyline
                        key={l.key}
                        fill="none"
                        stroke="currentColor"
                        className="text-zinc-200 dark:text-zinc-800"
                        strokeWidth={2}
                        points={l.pts}
                      />
                    ))}
                    {surfaceWireframe.bLines.map((l) => (
                      <polyline
                        key={l.key}
                        fill="none"
                        stroke="currentColor"
                        className="text-zinc-200 dark:text-zinc-800"
                        strokeWidth={2}
                        points={l.pts}
                      />
                    ))}

                    <defs>
                      <marker
                        id="surfaceArrow"
                        viewBox="0 0 10 10"
                        refX="8"
                        refY="5"
                        markerWidth="6"
                        markerHeight="6"
                        orient="auto-start-reverse"
                      >
                        <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
                      </marker>
                    </defs>
                    <circle
                      cx={surfaceWireframe.p0.x}
                      cy={surfaceWireframe.p0.y}
                      r={5}
                      fill="currentColor"
                      className="text-emerald-600 dark:text-emerald-400"
                    />
                    <line
                      x1={surfaceWireframe.p0.x}
                      y1={surfaceWireframe.p0.y}
                      x2={surfaceWireframe.p1.x}
                      y2={surfaceWireframe.p1.y}
                      stroke="currentColor"
                      className="text-emerald-600 dark:text-emerald-400"
                      strokeWidth={4}
                      markerEnd="url(#surfaceArrow)"
                      strokeLinecap="round"
                    />
                  </svg>
                ) : (
                  <div className="p-6 text-sm text-zinc-600 dark:text-zinc-400">
                    {points.length === 0 ? "No data" : "Surface unavailable"}
                  </div>
                )}
              </div>
            </div>

            <div className="grid gap-6 lg:grid-cols-2">
              {(() => {
                const makeSeries = (series: number[]) => {
                  const lw = 520;
                  const lh = 180;
                  const padding = 18;
                  const n = series.length;
                  const maxV = n > 0 ? Math.max(...series) : 1;
                  const minV = n > 0 ? Math.min(...series) : -1;
                  const span = Math.max(1e-9, maxV - minV);
                  const pts = series
                    .map((v, i) => {
                      const t = n === 1 ? 0 : i / (n - 1);
                      const x = padding + t * (lw - 2 * padding);
                      const y = padding + (1 - (v - minV) / span) * (lh - 2 * padding);
                      return { x, y };
                    });
                  const poly = pts.map((p) => `${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(" ");
                  const last = pts[pts.length - 1] ?? { x: padding, y: lh - padding };
                  return { lw, lh, padding, poly, last, minV, maxV };
                };

                const wS = makeSeries(wHistory.length > 0 ? wHistory : [w]);
                const bS = makeSeries(bHistory.length > 0 ? bHistory : [b]);

                const wDir = lastStep ? lastStep.wNext - lastStep.wPrev : 0;
                const bDir = lastStep ? lastStep.bNext - lastStep.bPrev : 0;

                const Card = ({
                  title,
                  value,
                  delta,
                  s,
                }: {
                  title: string;
                  value: number;
                  delta: number;
                  s: ReturnType<typeof makeSeries>;
                }) => (
                  <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
                    <div className="mb-2 flex items-baseline justify-between">
                      <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">{title}</h2>
                      <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                        {value.toFixed(4)} {delta === 0 ? "" : delta > 0 ? "↑" : "↓"}
                      </div>
                    </div>
                    <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
                      <svg viewBox={`0 0 ${s.lw} ${s.lh}`} className="h-auto w-full" role="img" aria-label={`${title} history`}>
                        <rect x={0} y={0} width={s.lw} height={s.lh} fill="transparent" />
                        <line
                          x1={s.padding}
                          y1={s.lh - s.padding}
                          x2={s.lw - s.padding}
                          y2={s.lh - s.padding}
                          stroke="currentColor"
                          className="text-zinc-200 dark:text-zinc-800"
                          strokeWidth={2}
                        />
                        <line
                          x1={s.padding}
                          y1={s.padding}
                          x2={s.padding}
                          y2={s.lh - s.padding}
                          stroke="currentColor"
                          className="text-zinc-200 dark:text-zinc-800"
                          strokeWidth={2}
                        />
                        <polyline
                          fill="none"
                          stroke="currentColor"
                          className="text-emerald-600 dark:text-emerald-400"
                          strokeWidth={3}
                          strokeLinejoin="round"
                          strokeLinecap="round"
                          points={s.poly}
                        />
                        <circle
                          cx={s.last.x}
                          cy={s.last.y}
                          r={4}
                          fill="currentColor"
                          className="text-emerald-600 dark:text-emerald-400"
                        />
                      </svg>
                    </div>
                    <div className="mt-2 font-mono text-[11px] text-zinc-500 dark:text-zinc-400">
                      {lastStep ? `t→t+1: ${delta.toFixed(6)}` : "t→t+1: -"}
                    </div>
                  </div>
                );

                return (
                  <>
                    <Card title="w (slope)" value={w} delta={wDir} s={wS} />
                    <Card title="b (bias)" value={b} delta={bDir} s={bS} />
                  </>
                );
              })()}
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
              <div className="mb-3 flex items-baseline justify-between">
                <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  圖形（資料點 + 目前模型線）
                </h2>
                <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                  x∈[{vp.xMin},{vp.xMax}], y∈[{vp.yMin},{vp.yMax}]
                </div>
              </div>
              <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
                <svg viewBox={`0 0 ${width} ${height}`} className="h-auto w-full" role="img" aria-label="model plot">
                  <rect x={0} y={0} width={width} height={height} fill="transparent" />

                  {gridXs.map((x) => {
                    const px = worldToSvg(x, 0, vp, width, height).x;
                    return (
                      <line
                        key={`gx-${x}`}
                        x1={px}
                        y1={0}
                        x2={px}
                        y2={height}
                        stroke="currentColor"
                        className="text-zinc-100 dark:text-zinc-900"
                        strokeWidth={2}
                      />
                    );
                  })}
                  {gridYs.map((y) => {
                    const py = worldToSvg(0, y, vp, width, height).y;
                    return (
                      <line
                        key={`gy-${y}`}
                        x1={0}
                        y1={py}
                        x2={width}
                        y2={py}
                        stroke="currentColor"
                        className="text-zinc-100 dark:text-zinc-900"
                        strokeWidth={2}
                      />
                    );
                  })}

                  <line
                    x1={axis.x0}
                    y1={0}
                    x2={axis.x0}
                    y2={height}
                    stroke="currentColor"
                    className="text-zinc-200 dark:text-zinc-800"
                    strokeWidth={3}
                  />
                  <line
                    x1={0}
                    y1={axis.y0}
                    x2={width}
                    y2={axis.y0}
                    stroke="currentColor"
                    className="text-zinc-200 dark:text-zinc-800"
                    strokeWidth={3}
                  />

                  {points.map((p, idx) => {
                    const sp = worldToSvg(p.x, p.y, vp, width, height);
                    return (
                      <circle
                        key={`p-${idx}`}
                        cx={sp.x}
                        cy={sp.y}
                        r={5}
                        fill="currentColor"
                        className="text-rose-600 dark:text-rose-400"
                        opacity={0.9}
                      />
                    );
                  })}

                  {/* Per-point error visualization: err_i = y_hat - y */}
                  {errorSegments.map((s, idx) => {
                    const pTrue = worldToSvg(s.x, s.yTrue, vp, width, height);
                    const pHat = worldToSvg(s.x, s.yHat, vp, width, height);
                    const isPos = s.err > 0;
                    return (
                      <g key={`e-${idx}`}>
                        <line
                          x1={pTrue.x}
                          y1={pTrue.y}
                          x2={pHat.x}
                          y2={pHat.y}
                          stroke="currentColor"
                          className={isPos ? "text-amber-600 dark:text-amber-400" : "text-sky-600 dark:text-sky-400"}
                          strokeWidth={3}
                          opacity={0.9}
                        />
                        <circle
                          cx={pHat.x}
                          cy={pHat.y}
                          r={4}
                          fill="none"
                          stroke="currentColor"
                          className={isPos ? "text-amber-600 dark:text-amber-400" : "text-sky-600 dark:text-sky-400"}
                          strokeWidth={2}
                        />
                      </g>
                    );
                  })}

                  <line
                    x1={line.p1.x}
                    y1={line.p1.y}
                    x2={line.p2.x}
                    y2={line.p2.y}
                    stroke="currentColor"
                    className="text-indigo-600 dark:text-indigo-400"
                    strokeWidth={4}
                    strokeLinecap="round"
                  />
                </svg>
              </div>
            </div>

            <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
              <div className="mb-3 flex items-baseline justify-between">
                <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                  Loss vs Epoch
                </h2>
                <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                  epochs: {lossHistory.length}
                </div>
              </div>
              <div className="mb-2 text-sm text-zinc-600 dark:text-zinc-400">
                這張圖是整體損失隨 epoch 的變化；你可以對照上方的 w/b 曲線，看參數怎麼走才讓 loss 下降。
              </div>
              <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
                {(() => {
                  const lw = 1100;
                  const lh = 220;
                  const padding = 18;
                  const n = lossHistory.length;
                  const maxLoss = n > 0 ? Math.max(...lossHistory) : 1;
                  const minLoss = n > 0 ? Math.min(...lossHistory) : 0;
                  const span = Math.max(1e-9, maxLoss - minLoss);

                  const pts = lossHistory
                    .map((loss, i) => {
                      const t = n === 1 ? 0 : i / (n - 1);
                      const x = padding + t * (lw - 2 * padding);
                      const y = padding + (1 - (loss - minLoss) / span) * (lh - 2 * padding);
                      return `${x.toFixed(2)},${y.toFixed(2)}`;
                    })
                    .join(" ");

                  return (
                    <svg viewBox={`0 0 ${lw} ${lh}`} className="h-auto w-full" role="img" aria-label="loss plot">
                      <rect x={0} y={0} width={lw} height={lh} fill="transparent" />
                      <line
                        x1={padding}
                        y1={lh - padding}
                        x2={lw - padding}
                        y2={lh - padding}
                        stroke="currentColor"
                        className="text-zinc-200 dark:text-zinc-800"
                        strokeWidth={2}
                      />
                      <line
                        x1={padding}
                        y1={padding}
                        x2={padding}
                        y2={lh - padding}
                        stroke="currentColor"
                        className="text-zinc-200 dark:text-zinc-800"
                        strokeWidth={2}
                      />
                      {n > 0 && (
                        <polyline
                          fill="none"
                          stroke="currentColor"
                          className="text-emerald-600 dark:text-emerald-400"
                          strokeWidth={3}
                          strokeLinejoin="round"
                          strokeLinecap="round"
                          points={pts}
                        />
                      )}
                    </svg>
                  );
                })()}
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
