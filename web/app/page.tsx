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
  const [lossHistory, setLossHistory] = useState<number[]>([]);
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

  const width = 900;
  const height = 520;
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

  async function trainGradientDescent() {
    if (isTraining || points.length === 0) return;
    setIsTraining(true);
    setLossHistory([]);
    setEpoch(0);
    setLastStep(null);

    const epochs = 60;
    const learningRate = 0.03;

    let currentW = w;
    let currentB = b;

    try {
      for (let i = 1; i <= epochs; i++) {
        const res = await fetch(`${API_BASE}/train/linear/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            w: currentW,
            b: currentB,
            learning_rate: learningRate,
            points,
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

        currentW = data.w;
        currentB = data.b;
        setW(data.w);
        setB(data.b);
        setMse(data.loss);
        setErrors(data.errors);
        setGradW(data.grad_w);
        setGradB(data.grad_b);
        setEpoch(i);
        setLossHistory((prev) => [...prev, data.loss]);
        setLastStep((prev) => {
          const wPrev = prev?.wNext ?? (i === 1 ? w : prev?.wPrev ?? w);
          const bPrev = prev?.bNext ?? (i === 1 ? b : prev?.bPrev ?? b);
          return {
            wPrev,
            bPrev,
            wNext: data.w,
            bNext: data.b,
            gradW: data.grad_w,
            gradB: data.grad_b,
            lr: learningRate,
            loss: data.loss,
          };
        });

        await sleep(80);
      }
    } finally {
      setIsTraining(false);
    }
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

  const lossStats = useMemo(() => {
    if (lossHistory.length === 0) return { current: null as number | null, best: null as number | null };
    const best = Math.min(...lossHistory);
    const current = lossHistory[lossHistory.length - 1];
    return { current, best };
  }, [lossHistory]);

  return (
    <div className="min-h-screen bg-zinc-50 font-sans text-zinc-900 dark:bg-black dark:text-zinc-50">
      <main className="mx-auto flex w-full max-w-5xl flex-col gap-8 px-6 py-10">
        <header className="flex flex-col gap-2">
          <h1 className="text-2xl font-semibold tracking-tight">線性模型視覺化</h1>
          <p className="text-sm text-zinc-600 dark:text-zinc-400">
            調整參數並觀察直線：y = wx + b
          </p>
        </header>

        <section className="grid gap-6 lg:grid-cols-[320px_1fr]">
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
              <div className="flex items-center justify-between gap-3">
                <button
                  type="button"
                  onClick={trainGradientDescent}
                  disabled={isTraining || isLoadingPoints || points.length === 0}
                  className="h-10 rounded-full bg-foreground px-4 text-sm font-medium text-background disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isTraining ? "Training..." : "Train (Gradient Descent)"}
                </button>
                <div className="font-mono text-xs text-zinc-600 dark:text-zinc-400">
                  epoch: {epoch}
                </div>
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
                  onChange={(e) => setW(parseFloat(e.target.value))}
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
                  onChange={(e) => setB(parseFloat(e.target.value))}
                  className="w-full"
                  disabled={isTraining}
                />
              </label>
            </div>
          </div>

          <div className="grid gap-6">
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
              <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
                {(() => {
                  const lw = 900;
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
