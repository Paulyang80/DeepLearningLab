"use client";

import { useMemo, useState } from "react";

function clamp(n: number, min: number, max: number) {
  return Math.max(min, Math.min(max, n));
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

  const width = 720;
  const height = 420;
  const vp: Viewport = { xMin: -5, xMax: 5, yMin: -5, yMax: 5 };

  const line = useMemo(() => {
    const x1 = vp.xMin;
    const y1 = w * x1 + b;
    const x2 = vp.xMax;
    const y2 = w * x2 + b;

    const p1 = worldToSvg(x1, clamp(y1, vp.yMin, vp.yMax), vp, width, height);
    const p2 = worldToSvg(x2, clamp(y2, vp.yMin, vp.yMax), vp, width, height);
    return { p1, p2 };
  }, [w, b]);

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
                  step={0.1}
                  value={w}
                  onChange={(e) => setW(Number(e.target.value))}
                  className="w-full"
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
                  step={0.1}
                  value={b}
                  onChange={(e) => setB(Number(e.target.value))}
                  className="w-full"
                />
              </label>
            </div>
          </div>

          <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
            <div className="mb-3 flex items-baseline justify-between">
              <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
                圖形
              </h2>
              <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
                x∈[{vp.xMin},{vp.xMax}], y∈[{vp.yMin},{vp.yMax}]
              </div>
            </div>

            <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
              <svg
                viewBox={`0 0 ${width} ${height}`}
                className="h-auto w-full"
                role="img"
                aria-label="line plot"
              >
                {/* grid */}
                {Array.from({ length: 11 }).map((_, i) => {
                  const t = i / 10;
                  const x = t * width;
                  const y = t * height;
                  return (
                    <g key={i}>
                      <line
                        x1={x}
                        y1={0}
                        x2={x}
                        y2={height}
                        stroke="currentColor"
                        className="text-zinc-200 dark:text-zinc-800"
                        strokeWidth={1}
                      />
                      <line
                        x1={0}
                        y1={y}
                        x2={width}
                        y2={y}
                        stroke="currentColor"
                        className="text-zinc-200 dark:text-zinc-800"
                        strokeWidth={1}
                      />
                    </g>
                  );
                })}

                {/* axes */}
                {(() => {
                  const xAxis = worldToSvg(0, 0, vp, width, height).y;
                  const yAxis = worldToSvg(0, 0, vp, width, height).x;
                  return (
                    <g>
                      <line
                        x1={0}
                        y1={xAxis}
                        x2={width}
                        y2={xAxis}
                        stroke="currentColor"
                        className="text-zinc-400 dark:text-zinc-600"
                        strokeWidth={2}
                      />
                      <line
                        x1={yAxis}
                        y1={0}
                        x2={yAxis}
                        y2={height}
                        stroke="currentColor"
                        className="text-zinc-400 dark:text-zinc-600"
                        strokeWidth={2}
                      />
                    </g>
                  );
                })()}

                {/* line */}
                <line
                  x1={line.p1.x}
                  y1={line.p1.y}
                  x2={line.p2.x}
                  y2={line.p2.y}
                  stroke="currentColor"
                  className="text-blue-600 dark:text-blue-400"
                  strokeWidth={4}
                  strokeLinecap="round"
                />
              </svg>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}
