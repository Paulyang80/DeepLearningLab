"use client";

import { useMemo, useState } from "react";

function clamp(v: number, min: number, max: number) {
  return Math.max(min, Math.min(max, v));
}

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
  // Box–Muller transform
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function mean(xs: number[]) {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

function variance(xs: number[], mu?: number) {
  if (xs.length === 0) return 0;
  const m = mu ?? mean(xs);
  let s = 0;
  for (const x of xs) {
    const d = x - m;
    s += d * d;
  }
  return s / xs.length;
}

function histogram(values: number[], bins: number, minX: number, maxX: number) {
  const counts = Array.from({ length: bins }, () => 0);
  const span = maxX - minX || 1;
  for (const v of values) {
    const t = (v - minX) / span;
    const idx = Math.floor(t * bins);
    if (idx < 0) continue;
    if (idx >= bins) continue;
    counts[idx] += 1;
  }
  const maxCount = Math.max(1, ...counts);
  return { counts, maxCount };
}

function HistogramSvg(props: {
  title: string;
  subtitle?: string;
  values: number[];
  bins: number;
  minX: number;
  maxX: number;
  colorClass: string;
}) {
  const width = 360;
  const height = 140;
  const pad = 10;
  const { counts, maxCount } = histogram(props.values, props.bins, props.minX, props.maxX);
  const barW = (width - pad * 2) / props.bins;

  return (
    <div className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mb-2 flex items-center justify-between gap-3">
        <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
          {props.title}
        </div>
        <div className="text-xs text-zinc-500 dark:text-zinc-400">
          {props.subtitle ?? `range [${format3(props.minX)}, ${format3(props.maxX)}]`}
        </div>
      </div>
      <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`} className="mx-auto block">
        <rect
          x={0}
          y={0}
          width={width}
          height={height}
          rx={14}
          className="fill-zinc-50 dark:fill-zinc-900/40"
        />
        {counts.map((c, i) => {
          const h = (c / maxCount) * (height - pad * 2);
          const x = pad + i * barW;
          const y = height - pad - h;
          return (
            <rect
              key={i}
              x={x}
              y={y}
              width={Math.max(0, barW - 1)}
              height={h}
              rx={2}
              className={props.colorClass}
              opacity={0.9}
            />
          );
        })}
      </svg>
      <div className="mt-2 text-xs text-zinc-600 dark:text-zinc-400">
        mean={format3(mean(props.values))}, var={format3(variance(props.values))}
      </div>
    </div>
  );
}

function format3(v: number) {
  return Number.isFinite(v) ? v.toFixed(3) : "-";
}

function SliderRow(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  right?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between text-xs text-zinc-600 dark:text-zinc-400">
        <div className="font-medium text-zinc-800 dark:text-zinc-200">
          {props.label}
        </div>
        <div className="font-mono">{props.right ?? props.value}</div>
      </div>
      <input
        type="range"
        min={props.min}
        max={props.max}
        step={props.step}
        value={props.value}
        onChange={(e) => props.onChange(Number(e.target.value))}
      />
    </div>
  );
}

function PillTabs<T extends string>(props: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (v: T) => void;
}) {
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-zinc-200 bg-white p-1 text-xs dark:border-zinc-800 dark:bg-zinc-950">
      {props.options.map((opt) => {
        const active = opt.value === props.value;
        return (
          <button
            key={opt.value}
            type="button"
            onClick={() => props.onChange(opt.value)}
            className={`rounded-full px-3 py-1.5 font-medium transition-colors ${
              active
                ? "bg-foreground text-background"
                : "text-zinc-700 hover:bg-zinc-50 dark:text-zinc-300 dark:hover:bg-zinc-900"
            }`}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

function Card(props: { title: string; subtitle?: string; children: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="mb-3 flex flex-col gap-1">
        <h2 className="text-sm font-medium text-zinc-900 dark:text-zinc-50">
          {props.title}
        </h2>
        {props.subtitle ? (
          <p className="text-xs text-zinc-600 dark:text-zinc-400">{props.subtitle}</p>
        ) : null}
      </div>
      {props.children}
    </section>
  );
}

function DropoutViz() {
  const [dropoutP, setDropoutP] = useState(0.5);
  const [seed, setSeed] = useState(7);

  const size = 280;
  const cols = 10;
  const rows = 10;
  const padding = 22;
  const cellW = (size - padding * 2) / (cols - 1);
  const cellH = (size - padding * 2) / (rows - 1);

  const q = 1 - dropoutP;

  const keptMask = useMemo(() => {
    const rng = mulberry32(seed);
    const mask: boolean[] = [];
    for (let i = 0; i < cols * rows; i++) {
      mask.push(rng() < q);
    }
    return mask;
  }, [seed, cols, rows, q]);

  const keptCount = keptMask.filter(Boolean).length;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">
          Dropout 會在訓練時隨機把一部分神經元輸出設為 0，迫使網路不要「太依賴某幾個特徵」，
          以降低 overfitting。
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
          <div className="font-medium text-zinc-900 dark:text-zinc-50">重點</div>
          <div className="mt-1 grid gap-1">
            <div>
              訓練：以機率 <span className="font-mono">p</span> 丟棄、以 <span className="font-mono">q=1-p</span> 保留
            </div>
            <div>
              推論：不丟棄（等效於把訓練時的隨機性拿掉）
            </div>
            <div>
              常用做法（Inverted Dropout）：訓練時把保留的輸出除以 <span className="font-mono">q</span>，
              讓期望值維持不變
            </div>
          </div>
        </div>

        <div className="grid gap-3">
          <SliderRow
            label="Dropout rate p"
            value={dropoutP}
            min={0}
            max={0.9}
            step={0.05}
            onChange={(v) => setDropoutP(clamp(v, 0, 0.9))}
            right={`${Math.round(dropoutP * 100)}%`}
          />
          <SliderRow
            label="Random seed"
            value={seed}
            min={1}
            max={99}
            step={1}
            onChange={(v) => setSeed(clamp(v, 1, 99))}
            right={`${seed}`}
          />
          <div className="text-xs text-zinc-600 dark:text-zinc-400">
            保留：<span className="font-mono">{keptCount}</span> / {cols * rows}（約{" "}
            <span className="font-mono">{Math.round((keptCount / (cols * rows)) * 100)}%</span>）
            ，Inverted scaling：<span className="font-mono">1/q = {q === 0 ? "∞" : (1 / q).toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
            訓練時：隨機遮罩（綠=保留，灰=丟棄）
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            q={format3(q)}
          </div>
        </div>

        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="mx-auto block"
        >
          <rect
            x={0}
            y={0}
            width={size}
            height={size}
            rx={16}
            className="fill-zinc-50 dark:fill-zinc-900/40"
          />
          {Array.from({ length: rows }).flatMap((_, r) =>
            Array.from({ length: cols }).map((_, c) => {
              const idx = r * cols + c;
              const kept = keptMask[idx];
              const cx = padding + c * cellW;
              const cy = padding + r * cellH;
              return (
                <circle
                  key={idx}
                  cx={cx}
                  cy={cy}
                  r={8}
                  className={
                    kept
                      ? "fill-emerald-500 dark:fill-emerald-400"
                      : "fill-zinc-200 dark:fill-zinc-700"
                  }
                  opacity={kept ? 1 : 0.55}
                />
              );
            }),
          )}
        </svg>

        <div className="mt-3 text-xs text-zinc-600 dark:text-zinc-400">
          直覺：每次 batch 都「換一張子網路」在學，最後像是做了很多模型的 ensemble。
        </div>
      </div>
    </div>
  );
}

function MomentumViz() {
  const [lr, setLr] = useState(0.12);
  const [beta, setBeta] = useState(0.9);
  const [steps, setSteps] = useState(25);
  const [w0, setW0] = useState(4);

  const sim = useMemo(() => {
    const history: { t: number; w: number; g: number; v: number }[] = [];
    let w = w0;
    let v = 0;

    for (let t = 1; t <= steps; t++) {
      const g = 2 * w; // d(w^2)/dw
      v = beta * v + g;
      w = w - lr * v;
      history.push({ t, w, g, v });
    }

    return history;
  }, [beta, lr, steps, w0]);

  const chartW = 420;
  const chartH = 180;

  const wValues = sim.map((d) => d.w).concat([w0]);
  const wMin = Math.min(...wValues, -0.5);
  const wMax = Math.max(...wValues, 0.5);

  function toXY(i: number, w: number) {
    const x = (i / Math.max(1, steps)) * chartW;
    const y = chartH - ((w - wMin) / (wMax - wMin || 1)) * chartH;
    return { x, y };
  }

  const points = (() => {
    const pts: string[] = [];
    // include t=0
    const p0 = toXY(0, w0);
    pts.push(`${p0.x},${p0.y}`);
    sim.forEach((d, idx) => {
      const p = toXY(idx + 1, d.w);
      pts.push(`${p.x},${p.y}`);
    });
    return pts.join(" ");
  })();

  const last = sim[sim.length - 1];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">
          Momentum（動量）把過去的梯度累積成「速度」<span className="font-mono">v</span>，
          讓更新方向更穩定、在長谷底加速、在雜訊梯度下更不抖。
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
          <div className="font-medium text-zinc-900 dark:text-zinc-50">更新式（示意）</div>
          <div className="mt-1 grid gap-1 font-mono">
            <div>g_t = ∂L/∂w</div>
            <div>v_t = β·v_(t-1) + g_t</div>
            <div>w_t = w_(t-1) − lr·v_t</div>
          </div>
        </div>

        <div className="grid gap-3">
          <SliderRow
            label="Learning rate"
            value={lr}
            min={0.02}
            max={0.3}
            step={0.01}
            onChange={(v) => setLr(clamp(v, 0.02, 0.3))}
            right={format3(lr)}
          />
          <SliderRow
            label="Momentum β"
            value={beta}
            min={0}
            max={0.99}
            step={0.01}
            onChange={(v) => setBeta(clamp(v, 0, 0.99))}
            right={format3(beta)}
          />
          <SliderRow
            label="Steps"
            value={steps}
            min={5}
            max={60}
            step={1}
            onChange={(v) => setSteps(clamp(v, 5, 60))}
            right={`${steps}`}
          />
          <SliderRow
            label="Initial w0"
            value={w0}
            min={-6}
            max={6}
            step={0.5}
            onChange={(v) => setW0(clamp(v, -6, 6))}
            right={format3(w0)}
          />
        </div>

        <div className="text-xs text-zinc-600 dark:text-zinc-400">
          最後一步：w={format3(last?.w ?? w0)}，g={format3(last?.g ?? 0)}，v={format3(last?.v ?? 0)}
        </div>
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
            例：最小化 L(w)=w²（往 0 收斂）
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            w 範圍 [{format3(wMin)},{format3(wMax)}]
          </div>
        </div>

        <svg
          width={chartW}
          height={chartH}
          viewBox={`0 0 ${chartW} ${chartH}`}
          className="mx-auto block"
        >
          <rect
            x={0}
            y={0}
            width={chartW}
            height={chartH}
            rx={14}
            className="fill-zinc-50 dark:fill-zinc-900/40"
          />

          {/* baseline at w=0 */}
          <line
            x1={0}
            x2={chartW}
            y1={toXY(0, 0).y}
            y2={toXY(0, 0).y}
            className="stroke-zinc-200 dark:stroke-zinc-700"
          />

          <polyline
            points={points}
            fill="none"
            className="stroke-emerald-500 dark:stroke-emerald-400"
            strokeWidth={3}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {/* last point */}
          {(() => {
            const p = toXY(steps, last?.w ?? w0);
            return (
              <circle
                cx={p.x}
                cy={p.y}
                r={5}
                className="fill-emerald-600 dark:fill-emerald-400"
              />
            );
          })()}
        </svg>

        <div className="mt-3 text-xs text-zinc-600 dark:text-zinc-400">
          β 越大，更新越「看重歷史方向」；lr 太大會 overshoot。
        </div>
      </div>
    </div>
  );
}

function AdamViz() {
  const [lr, setLr] = useState(0.2);
  const [beta1, setBeta1] = useState(0.9);
  const [beta2, setBeta2] = useState(0.999);
  const [epsExp, setEpsExp] = useState(-8); // eps = 10^exp
  const [noise, setNoise] = useState(0.3);
  const [steps, setSteps] = useState(30);
  const [seed, setSeed] = useState(12);
  const [w0, setW0] = useState(4);

  const eps = Math.pow(10, epsExp);

  const sim = useMemo(() => {
    const rng = mulberry32(seed);
    const history: {
      t: number;
      w: number;
      g: number;
      m: number;
      v: number;
      mHat: number;
      vHat: number;
      step: number;
    }[] = [];

    let w = w0;
    let m = 0;
    let v = 0;

    for (let t = 1; t <= steps; t++) {
      const gClean = 2 * w;
      const gNoise = (rng() * 2 - 1) * noise;
      const g = gClean + gNoise;

      m = beta1 * m + (1 - beta1) * g;
      v = beta2 * v + (1 - beta2) * (g * g);

      const mHat = m / (1 - Math.pow(beta1, t));
      const vHat = v / (1 - Math.pow(beta2, t));

      const denom = Math.sqrt(vHat) + eps;
      const stepSize = denom === 0 ? 0 : (lr * mHat) / denom;
      w = w - stepSize;

      history.push({ t, w, g, m, v, mHat, vHat, step: stepSize });
    }

    return history;
  }, [beta1, beta2, eps, lr, noise, seed, steps, w0]);

  const chartW = 420;
  const chartH = 180;

  const wValues = sim.map((d) => d.w).concat([w0]);
  const wMin = Math.min(...wValues, -0.5);
  const wMax = Math.max(...wValues, 0.5);

  function toXY(i: number, w: number) {
    const x = (i / Math.max(1, steps)) * chartW;
    const y = chartH - ((w - wMin) / (wMax - wMin || 1)) * chartH;
    return { x, y };
  }

  const points = (() => {
    const pts: string[] = [];
    const p0 = toXY(0, w0);
    pts.push(`${p0.x},${p0.y}`);
    sim.forEach((d, idx) => {
      const p = toXY(idx + 1, d.w);
      pts.push(`${p.x},${p.y}`);
    });
    return pts.join(" ");
  })();

  const last = sim[sim.length - 1];

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">
          Adam 把 Momentum（一次動量）+ RMSProp（二次動量/尺度）合在一起：
          用 <span className="font-mono">m</span> 追蹤梯度平均、用 <span className="font-mono">v</span> 追蹤梯度平方平均，
          讓每個參數自適應步長。
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
          <div className="font-medium text-zinc-900 dark:text-zinc-50">更新式（示意）</div>
          <div className="mt-1 grid gap-1 font-mono">
            <div>m_t = β1·m_(t-1) + (1-β1)·g_t</div>
            <div>v_t = β2·v_(t-1) + (1-β2)·g_t²</div>
            <div>m̂_t = m_t / (1-β1^t), v̂_t = v_t / (1-β2^t)</div>
            <div>w_t = w_(t-1) − lr · m̂_t / (sqrt(v̂_t)+ε)</div>
          </div>
          <div className="mt-2 text-xs text-zinc-600 dark:text-zinc-400">
            為什麼要 bias correction？因為 m、v 從 0 開始，前幾步會被 0 拉小（特別是 β 接近 1 時），
            直接用 m_t、v_t 會低估「真實的一階/二階動量」。用除以 (1-β^t) 把這個初始化偏差校正回來，
            讓早期步長不會異常保守。
          </div>
        </div>

        <div className="grid gap-3">
          <SliderRow
            label="Learning rate"
            value={lr}
            min={0.02}
            max={0.6}
            step={0.01}
            onChange={(v) => setLr(clamp(v, 0.02, 0.6))}
            right={format3(lr)}
          />
          <SliderRow
            label="β1 (momentum)"
            value={beta1}
            min={0}
            max={0.99}
            step={0.01}
            onChange={(v) => setBeta1(clamp(v, 0, 0.99))}
            right={format3(beta1)}
          />
          <SliderRow
            label="β2 (RMS)"
            value={beta2}
            min={0.9}
            max={0.999}
            step={0.001}
            onChange={(v) => setBeta2(clamp(v, 0.9, 0.999))}
            right={beta2.toFixed(3)}
          />
          <SliderRow
            label="ε = 10^k"
            value={epsExp}
            min={-10}
            max={-2}
            step={1}
            onChange={(v) => setEpsExp(clamp(v, -10, -2))}
            right={`10^${epsExp} = ${eps.toExponential(0)}`}
          />
          <SliderRow
            label="Gradient noise"
            value={noise}
            min={0}
            max={1.0}
            step={0.05}
            onChange={(v) => setNoise(clamp(v, 0, 1))}
            right={format3(noise)}
          />
          <SliderRow
            label="Steps"
            value={steps}
            min={10}
            max={80}
            step={1}
            onChange={(v) => setSteps(clamp(v, 10, 80))}
            right={`${steps}`}
          />
          <SliderRow
            label="Seed"
            value={seed}
            min={1}
            max={99}
            step={1}
            onChange={(v) => setSeed(clamp(v, 1, 99))}
            right={`${seed}`}
          />
          <SliderRow
            label="Initial w0"
            value={w0}
            min={-6}
            max={6}
            step={0.5}
            onChange={(v) => setW0(clamp(v, -6, 6))}
            right={format3(w0)}
          />
        </div>

        <div className="text-xs text-zinc-600 dark:text-zinc-400">
          最後一步：w={format3(last?.w ?? w0)}，g={format3(last?.g ?? 0)}，m̂={format3(last?.mHat ?? 0)}，sqrt(v̂)={format3(Math.sqrt(last?.vHat ?? 0))}
        </div>
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
            例：帶雜訊梯度下最小化 L(w)=w²
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            w 範圍 [{format3(wMin)},{format3(wMax)}]
          </div>
        </div>

        <svg
          width={chartW}
          height={chartH}
          viewBox={`0 0 ${chartW} ${chartH}`}
          className="mx-auto block"
        >
          <rect
            x={0}
            y={0}
            width={chartW}
            height={chartH}
            rx={14}
            className="fill-zinc-50 dark:fill-zinc-900/40"
          />

          <line
            x1={0}
            x2={chartW}
            y1={toXY(0, 0).y}
            y2={toXY(0, 0).y}
            className="stroke-zinc-200 dark:stroke-zinc-700"
          />

          <polyline
            points={points}
            fill="none"
            className="stroke-sky-500 dark:stroke-sky-400"
            strokeWidth={3}
            strokeLinejoin="round"
            strokeLinecap="round"
          />

          {(() => {
            const p = toXY(steps, last?.w ?? w0);
            return (
              <circle
                cx={p.x}
                cy={p.y}
                r={5}
                className="fill-sky-600 dark:fill-sky-400"
              />
            );
          })()}
        </svg>

        <div className="mt-3 text-xs text-zinc-600 dark:text-zinc-400">
          觀察：雜訊增加時，Adam 仍能靠 <span className="font-mono">sqrt(v̂)</span> 調整步長，讓收斂更穩。
        </div>
      </div>
    </div>
  );
}

function BatchNormViz() {
  const [mode, setMode] = useState<"train" | "infer">("train");
  const [batchSize, setBatchSize] = useState(32);
  const [mu, setMu] = useState(1.2);
  const [sigma, setSigma] = useState(1.0);
  const [gamma, setGamma] = useState(1.0);
  const [beta, setBeta] = useState(0.0);
  const [epsExp, setEpsExp] = useState(-5);
  const [momentum, setMomentum] = useState(0.1);
  const [warmupBatches, setWarmupBatches] = useState(20);
  const [seed, setSeed] = useState(11);

  const eps = Math.pow(10, epsExp);

  const sim = useMemo(() => {
    const rng = mulberry32(seed);

    let runningMean = 0;
    let runningVar = 1;

    for (let k = 0; k < warmupBatches; k++) {
      const xs: number[] = [];
      for (let i = 0; i < batchSize; i++) {
        xs.push(mu + sigma * randn(rng));
      }
      const m = mean(xs);
      const v = variance(xs, m);
      runningMean = (1 - momentum) * runningMean + momentum * m;
      runningVar = (1 - momentum) * runningVar + momentum * v;
    }

    const x: number[] = [];
    for (let i = 0; i < batchSize; i++) {
      x.push(mu + sigma * randn(rng));
    }

    const batchMean = mean(x);
    const batchVar = variance(x, batchMean);

    const normMean = mode === "train" ? batchMean : runningMean;
    const normVar = mode === "train" ? batchVar : runningVar;

    const xHat = x.map((v) => (v - normMean) / Math.sqrt(normVar + eps));
    const y = xHat.map((v) => gamma * v + beta);

    return {
      x,
      xHat,
      y,
      batchMean,
      batchVar,
      runningMean,
      runningVar,
      normMean,
      normVar,
    };
  }, [batchSize, beta, eps, gamma, mode, momentum, mu, seed, sigma, warmupBatches]);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">
          Batch Normalization 會把同一個 batch 的 activations 做標準化（變成接近 mean=0、var=1），再用
          <span className="font-mono">γ</span>、<span className="font-mono">β</span> 做回縮放與平移。
          訓練時用 batch 統計量；推論時改用 running mean/var（比較穩）。
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
          <div className="font-medium text-zinc-900 dark:text-zinc-50">公式（示意）</div>
          <div className="mt-1 grid gap-1 font-mono">
            <div>x̂ = (x − mean) / sqrt(var + ε)</div>
            <div>y = γ·x̂ + β</div>
          </div>
          <div className="mt-2 text-xs text-zinc-600 dark:text-zinc-400">
            training：mean/var 取自當前 batch；inference：mean/var 取自 running statistics。
          </div>
        </div>

        <div className="flex items-center justify-between">
          <PillTabs
            value={mode}
            options={[
              { value: "train", label: "Training (batch stats)" },
              { value: "infer", label: "Inference (running stats)" },
            ]}
            onChange={setMode}
          />
        </div>

        <div className="grid gap-3">
          <SliderRow
            label="Batch size"
            value={batchSize}
            min={4}
            max={128}
            step={4}
            onChange={(v) => setBatchSize(clamp(v, 4, 128))}
            right={`${batchSize}`}
          />
          <SliderRow
            label="Input mean μ"
            value={mu}
            min={-2}
            max={2}
            step={0.1}
            onChange={(v) => setMu(clamp(v, -2, 2))}
            right={format3(mu)}
          />
          <SliderRow
            label="Input std σ"
            value={sigma}
            min={0.2}
            max={2.5}
            step={0.1}
            onChange={(v) => setSigma(clamp(v, 0.2, 2.5))}
            right={format3(sigma)}
          />
          <SliderRow
            label="γ (scale)"
            value={gamma}
            min={0.2}
            max={2.5}
            step={0.1}
            onChange={(v) => setGamma(clamp(v, 0.2, 2.5))}
            right={format3(gamma)}
          />
          <SliderRow
            label="β (shift)"
            value={beta}
            min={-2}
            max={2}
            step={0.1}
            onChange={(v) => setBeta(clamp(v, -2, 2))}
            right={format3(beta)}
          />
          <SliderRow
            label="ε = 10^k"
            value={epsExp}
            min={-8}
            max={-2}
            step={1}
            onChange={(v) => setEpsExp(clamp(v, -8, -2))}
            right={`10^${epsExp} = ${eps.toExponential(0)}`}
          />
          <SliderRow
            label="Running momentum"
            value={momentum}
            min={0.01}
            max={0.5}
            step={0.01}
            onChange={(v) => setMomentum(clamp(v, 0.01, 0.5))}
            right={format3(momentum)}
          />
          <SliderRow
            label="Warmup batches (for running stats)"
            value={warmupBatches}
            min={0}
            max={80}
            step={1}
            onChange={(v) => setWarmupBatches(clamp(v, 0, 80))}
            right={`${warmupBatches}`}
          />
          <SliderRow
            label="Seed"
            value={seed}
            min={1}
            max={99}
            step={1}
            onChange={(v) => setSeed(clamp(v, 1, 99))}
            right={`${seed}`}
          />
        </div>

        <div className="text-xs text-zinc-600 dark:text-zinc-400">
          batch mean/var: {format3(sim.batchMean)} / {format3(sim.batchVar)} ・ running mean/var: {format3(sim.runningMean)} / {format3(sim.runningVar)} ・ used mean/var: {format3(sim.normMean)} / {format3(sim.normVar)}
        </div>
      </div>

      <div className="grid gap-3">
        <HistogramSvg
          title="x (before)"
          values={sim.x}
          bins={18}
          minX={mu - 4 * sigma}
          maxX={mu + 4 * sigma}
          colorClass="fill-zinc-300 dark:fill-zinc-600"
        />
        <HistogramSvg
          title="x̂ (normalized)"
          values={sim.xHat}
          bins={18}
          minX={-4}
          maxX={4}
          colorClass="fill-sky-500 dark:fill-sky-400"
        />
        <HistogramSvg
          title="y = γ·x̂ + β"
          values={sim.y}
          bins={18}
          minX={beta - 4 * gamma}
          maxX={beta + 4 * gamma}
          colorClass="fill-emerald-500 dark:fill-emerald-400"
        />
      </div>
    </div>
  );
}

function WeightInitViz() {
  const [init, setInit] = useState<"xavier" | "he" | "naive">("xavier");
  const [activation, setActivation] = useState<"linear" | "relu">("relu");
  const [fanIn, setFanIn] = useState(64);
  const [fanOut, setFanOut] = useState(64);
  const [samples, setSamples] = useState(600);
  const [seed, setSeed] = useState(21);

  const sim = useMemo(() => {
    const rng = mulberry32(seed);

    const fi = Math.max(1, Math.floor(fanIn));
    const fo = Math.max(1, Math.floor(fanOut));
    const n = Math.max(50, Math.floor(samples));

    let wStd = 1;
    if (init === "xavier") wStd = Math.sqrt(2 / (fi + fo));
    if (init === "he") wStd = Math.sqrt(2 / fi);

    // weights for a single neuron: w ~ N(0, wStd^2)
    const z: number[] = [];
    const a: number[] = [];

    for (let s = 0; s < n; s++) {
      let sum = 0;
      for (let i = 0; i < fi; i++) {
        const x = randn(rng); // Var ~ 1
        const w = wStd * randn(rng);
        sum += w * x;
      }
      const pre = sum;
      z.push(pre);
      const post = activation === "relu" ? Math.max(0, pre) : pre;
      a.push(post);
    }

    return { wStd, z, a };
  }, [activation, fanIn, fanOut, init, samples, seed]);

  const zVar = variance(sim.z);
  const aVar = variance(sim.a);

  const zRange = 5;
  const aRange = 5;

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">
          Weight Initialization 想要讓訊號在層與層之間不要爆炸/消失。
          Xavier（對稱激活/線性）與 He（ReLU）會用 fan-in / fan-out 去設定權重方差。
        </div>

        <div className="flex items-center justify-between gap-3">
          <PillTabs
            value={init}
            options={[
              { value: "xavier", label: "Xavier" },
              { value: "he", label: "He" },
              { value: "naive", label: "Naive (std=1)" },
            ]}
            onChange={setInit}
          />
        </div>

        <div className="flex items-center justify-between gap-3">
          <PillTabs
            value={activation}
            options={[
              { value: "relu", label: "ReLU" },
              { value: "linear", label: "Linear" },
            ]}
            onChange={setActivation}
          />
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
          <div className="font-medium text-zinc-900 dark:text-zinc-50">常見標準差（normal init）</div>
          <div className="mt-1 grid gap-1 font-mono">
            <div>Xavier: std = sqrt(2 / (fan_in + fan_out))</div>
            <div>He: std = sqrt(2 / fan_in)</div>
          </div>
        </div>

        <div className="grid gap-3">
          <SliderRow
            label="fan_in"
            value={fanIn}
            min={4}
            max={512}
            step={4}
            onChange={(v) => setFanIn(clamp(v, 4, 512))}
            right={`${fanIn}`}
          />
          <SliderRow
            label="fan_out"
            value={fanOut}
            min={4}
            max={512}
            step={4}
            onChange={(v) => setFanOut(clamp(v, 4, 512))}
            right={`${fanOut}`}
          />
          <SliderRow
            label="Samples"
            value={samples}
            min={100}
            max={2000}
            step={50}
            onChange={(v) => setSamples(clamp(v, 100, 2000))}
            right={`${samples}`}
          />
          <SliderRow
            label="Seed"
            value={seed}
            min={1}
            max={99}
            step={1}
            onChange={(v) => setSeed(clamp(v, 1, 99))}
            right={`${seed}`}
          />
        </div>

        <div className="text-xs text-zinc-600 dark:text-zinc-400">
          w std={format3(sim.wStd)} ・ pre-activation var={format3(zVar)} ・ post-activation var={format3(aVar)}
        </div>
      </div>

      <div className="grid gap-3">
        <HistogramSvg
          title="z = W·x (pre-activation)"
          values={sim.z}
          bins={18}
          minX={-zRange}
          maxX={zRange}
          colorClass="fill-sky-500 dark:fill-sky-400"
        />
        <HistogramSvg
          title={activation === "relu" ? "a = ReLU(z)" : "a = z"}
          values={sim.a}
          bins={18}
          minX={activation === "relu" ? 0 : -aRange}
          maxX={aRange}
          colorClass="fill-emerald-500 dark:fill-emerald-400"
        />
        <div className="text-xs text-zinc-600 dark:text-zinc-400">
          直覺：如果 z 的方差層層變大/變小，就會導致梯度爆炸或消失；Xavier/He 的目標是讓尺度更穩。
        </div>
      </div>
    </div>
  );
}

function RegularizationViz() {
  const [mode, setMode] = useState<"l1" | "l2">("l2");
  const [lambda, setLambda] = useState(0.3);
  const [wStar, setWStar] = useState(2.0);
  const [w, setW] = useState(1.0);

  const chartW = 520;
  const chartH = 220;

  const xs = useMemo(() => {
    const arr: number[] = [];
    const n = 120;
    const xMin = -4;
    const xMax = 4;
    for (let i = 0; i <= n; i++) {
      arr.push(xMin + (i / n) * (xMax - xMin));
    }
    return arr;
  }, []);

  const curves = useMemo(() => {
    const dataLoss = xs.map((x) => (x - wStar) * (x - wStar));
    const penalty = xs.map((x) =>
      mode === "l2" ? lambda * (x * x) : lambda * Math.abs(x),
    );
    const total = dataLoss.map((v, i) => v + penalty[i]);

    const yMax = Math.max(...total);
    return { dataLoss, penalty, total, yMax };
  }, [lambda, mode, wStar, xs]);

  function toXY(x: number, y: number) {
    const xMin = -4;
    const xMax = 4;
    const px = ((x - xMin) / (xMax - xMin)) * chartW;
    const py = chartH - (y / (curves.yMax || 1)) * chartH;
    return { x: px, y: py };
  }

  const poly = (() => {
    const toPoints = (ys: number[]) =>
      ys
        .map((y, i) => {
          const p = toXY(xs[i], y);
          return `${p.x},${p.y}`;
        })
        .join(" ");

    return {
      data: toPoints(curves.dataLoss),
      pen: toPoints(curves.penalty),
      total: toPoints(curves.total),
    };
  })();

  const penaltyGrad = useMemo(() => {
    if (mode === "l2") {
      return 2 * lambda * w;
    }
    // L1 subgradient: sign(w) (0 is ambiguous)
    if (w > 0) return lambda;
    if (w < 0) return -lambda;
    return 0;
  }, [lambda, mode, w]);

  const dataGrad = 2 * (w - wStar);
  const totalGrad = dataGrad + penaltyGrad;

  const marker = (() => {
    const data = (w - wStar) * (w - wStar);
    const pen = mode === "l2" ? lambda * w * w : lambda * Math.abs(w);
    const tot = data + pen;

    const p = toXY(w, tot);
    return { data, pen, tot, p };
  })();

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <div className="text-sm text-zinc-600 dark:text-zinc-400">
          Regularization 會在 loss 上加一項「偏好簡單模型」的懲罰，讓權重不要長太大。
          常見的是 L2（Weight Decay）與 L1（促進稀疏）。
        </div>

        <div className="flex items-center justify-between gap-3">
          <PillTabs
            value={mode}
            options={[
              { value: "l2", label: "L2 / Weight Decay" },
              { value: "l1", label: "L1 / Sparsity" },
            ]}
            onChange={setMode}
          />
        </div>

        <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
          <div className="font-medium text-zinc-900 dark:text-zinc-50">示意 loss</div>
          <div className="mt-1 grid gap-1">
            <div>
              data loss：<span className="font-mono">(w - w*)²</span>（想要靠近真值 w*）
            </div>
            <div>
              penalty：<span className="font-mono">λ·||w||</span> 或 <span className="font-mono">λ·||w||²</span>
            </div>
            <div>
              total：<span className="font-mono">(w - w*)² + penalty</span>
            </div>
          </div>
        </div>

        <div className="grid gap-3">
          <SliderRow
            label="λ (regularization strength)"
            value={lambda}
            min={0}
            max={1.5}
            step={0.05}
            onChange={(v) => setLambda(clamp(v, 0, 1.5))}
            right={format3(lambda)}
          />
          <SliderRow
            label="w* (target)"
            value={wStar}
            min={-3}
            max={3}
            step={0.1}
            onChange={(v) => setWStar(clamp(v, -3, 3))}
            right={format3(wStar)}
          />
          <SliderRow
            label="current w"
            value={w}
            min={-4}
            max={4}
            step={0.1}
            onChange={(v) => setW(clamp(v, -4, 4))}
            right={format3(w)}
          />
        </div>

        <div className="text-xs text-zinc-600 dark:text-zinc-400">
          梯度：data={format3(dataGrad)}，penalty={format3(penaltyGrad)}，total={format3(totalGrad)}
        </div>
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-950">
        <div className="mb-2 flex items-center justify-between">
          <div className="text-xs font-medium text-zinc-700 dark:text-zinc-300">
            曲線：data（灰）、penalty（紫）、total（綠）
          </div>
          <div className="text-xs text-zinc-500 dark:text-zinc-400">
            marker: total={format3(marker.tot)}
          </div>
        </div>

        <svg
          width={chartW}
          height={chartH}
          viewBox={`0 0 ${chartW} ${chartH}`}
          className="mx-auto block"
        >
          <rect
            x={0}
            y={0}
            width={chartW}
            height={chartH}
            rx={14}
            className="fill-zinc-50 dark:fill-zinc-900/40"
          />

          {/* x axis */}
          <line
            x1={0}
            x2={chartW}
            y1={chartH - 1}
            y2={chartH - 1}
            className="stroke-zinc-200 dark:stroke-zinc-700"
          />

          <polyline
            points={poly.data}
            fill="none"
            className="stroke-zinc-300 dark:stroke-zinc-600"
            strokeWidth={2}
          />
          <polyline
            points={poly.pen}
            fill="none"
            className="stroke-violet-500 dark:stroke-violet-400"
            strokeWidth={2}
          />
          <polyline
            points={poly.total}
            fill="none"
            className="stroke-emerald-500 dark:stroke-emerald-400"
            strokeWidth={3}
          />

          {/* marker */}
          <circle
            cx={marker.p.x}
            cy={marker.p.y}
            r={5}
            className="fill-emerald-600 dark:fill-emerald-400"
          />

          {/* gradient arrow (1D): show direction to decrease total */}
          {(() => {
            const dir = totalGrad === 0 ? 0 : totalGrad > 0 ? -1 : 1;
            const dx = dir * 40;
            const y = marker.p.y;
            const x1 = marker.p.x;
            const x2 = clamp(x1 + dx, 0, chartW);
            return (
              <g>
                <line
                  x1={x1}
                  y1={y}
                  x2={x2}
                  y2={y}
                  className="stroke-zinc-700 dark:stroke-zinc-200"
                  strokeWidth={2}
                />
                <circle
                  cx={x2}
                  cy={y}
                  r={3}
                  className="fill-zinc-700 dark:fill-zinc-200"
                />
              </g>
            );
          })()}
        </svg>

        <div className="mt-3 text-xs text-zinc-600 dark:text-zinc-400">
          L2：梯度與 w 成正比（越大拉越回來）。L1：在 0 附近有尖角，容易把小權重推到 0（稀疏）。
        </div>
      </div>
    </div>
  );
}

export default function TechniquesPage() {
  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-8 py-10">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Techniques: Dropout / Momentum / Adam / Regularization
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          這一頁用小型互動視覺化，幫你建立直覺：每個技巧「在做什麼」、
          為什麼能讓訓練更穩或泛化更好。
        </p>
      </div>

      <Card
        title="Dropout"
        subtitle="訓練時隨機丟棄神經元，降低 co-adaptation（共適應）"
      >
        <DropoutViz />
      </Card>

      <Card
        title="Momentum (SGD + Momentum)"
        subtitle="用速度累積梯度方向，加速並降低震盪"
      >
        <MomentumViz />
      </Card>

      <Card
        title="Adam"
        subtitle="一階動量 + 二階尺度（自適應步長），對雜訊與不同尺度較穩"
      >
        <AdamViz />
      </Card>

      <Card
        title="Batch Normalization"
        subtitle="用 batch 統計量做標準化；推論時用 running mean/var"
      >
        <BatchNormViz />
      </Card>

      <Card
        title="Weight Initialization (Xavier / He)"
        subtitle="用 fan-in / fan-out 控制權重方差，避免訊號尺度爆炸或消失"
      >
        <WeightInitViz />
      </Card>

      <Card
        title="L1 / L2 Regularization"
        subtitle="在 loss 加 penalty：偏好小權重、抑制過擬合"
      >
        <RegularizationViz />
      </Card>

      <div className="text-xs text-zinc-500 dark:text-zinc-400">
        提示：這些視覺化是「直覺模型」，用來理解更新方向與趨勢；真正的深度網路會在高維空間裡進行。
      </div>
    </main>
  );
}
