"use client";

import { useEffect, useMemo, useState } from "react";

type Point = { x: number; y: number };

type DatasetResponse = { points: Point[] };

type ExplainLogisticResponse = {
  loss: number;
  grad_w: number;
  grad_b: number;
  probs: number[];
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

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

export default function LogisticPage() {
  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE?.trim() || "http://localhost:8000";

  const [w, setW] = useState<number>(1);
  const [b, setB] = useState<number>(0);
  const [activation, setActivation] = useState<"sigmoid" | "tanh" | "linear">(
    "sigmoid",
  );

  const [points, setPoints] = useState<Point[]>([]);
  const [isLoadingPoints, setIsLoadingPoints] = useState<boolean>(true);

  const [loss, setLoss] = useState<number>(0);
  const [gradW, setGradW] = useState<number>(0);
  const [gradB, setGradB] = useState<number>(0);
  const [probs, setProbs] = useState<number[]>([]);

  const [epoch, setEpoch] = useState<number>(0);
  const [isTraining, setIsTraining] = useState<boolean>(false);

  const learningRate = 0.2;

  useEffect(() => {
    const controller = new AbortController();
    async function loadDataset() {
      setIsLoadingPoints(true);
      try {
        const res = await fetch(`${API_BASE}/dataset/logistic_1d`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            n: 60,
            seed: 7,
            x_min: -5,
            x_max: 5,
            true_w: 1.2,
            true_b: -0.3,
          }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`dataset error: ${res.status}`);
        const data = (await res.json()) as DatasetResponse;
        setPoints(data.points);
      } catch {
        setPoints([]);
      } finally {
        setIsLoadingPoints(false);
      }
    }

    void loadDataset();
    return () => controller.abort();
  }, [API_BASE]);

  useEffect(() => {
    if (points.length === 0) {
      setLoss(0);
      setGradW(0);
      setGradB(0);
      setProbs([]);
      return;
    }

    if (isTraining) return;

    const controller = new AbortController();
    async function loadExplain() {
      try {
        const res = await fetch(`${API_BASE}/explain/logistic_1d`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ w, b, activation, points }),
          signal: controller.signal,
        });
        if (!res.ok) throw new Error(`explain error: ${res.status}`);
        const data = (await res.json()) as ExplainLogisticResponse;
        setLoss(data.loss);
        setGradW(data.grad_w);
        setGradB(data.grad_b);
        setProbs(data.probs);
      } catch {
        setLoss(0);
        setGradW(0);
        setGradB(0);
        setProbs([]);
      }
    }

    void loadExplain();
    return () => controller.abort();
  }, [API_BASE, points, w, b, activation, isTraining]);

  async function runTrainStep() {
    if (isTraining || isLoadingPoints || points.length === 0) return;
    setIsTraining(true);
    try {
      const res = await fetch(`${API_BASE}/train/logistic_1d/step`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          w,
          b,
          activation,
          learning_rate: learningRate,
          points,
        }),
      });
      if (!res.ok) throw new Error(`train error: ${res.status}`);
      const data = (await res.json()) as ExplainLogisticResponse & {
        w: number;
        b: number;
      };

      setW(data.w);
      setB(data.b);
      setLoss(data.loss);
      setGradW(data.grad_w);
      setGradB(data.grad_b);
      setProbs(data.probs);
      setEpoch((prev) => prev + 1);
    } finally {
      setIsTraining(false);
    }
  }

  async function autoTrain() {
    if (isTraining || isLoadingPoints || points.length === 0) return;
    setIsTraining(true);
    try {
      setEpoch(0);
      let currentW = w;
      let currentB = b;
      for (let i = 0; i < 40; i++) {
        const res = await fetch(`${API_BASE}/train/logistic_1d/step`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            w: currentW,
            b: currentB,
            activation,
            learning_rate: learningRate,
            points,
          }),
        });
        if (!res.ok) throw new Error(`train error: ${res.status}`);
        const data = (await res.json()) as ExplainLogisticResponse & {
          w: number;
          b: number;
        };
        currentW = data.w;
        currentB = data.b;
        setW(data.w);
        setB(data.b);
        setLoss(data.loss);
        setGradW(data.grad_w);
        setGradB(data.grad_b);
        setProbs(data.probs);
        setEpoch((prev) => prev + 1);
        await sleep(60);
      }
    } finally {
      setIsTraining(false);
    }
  }

  const vp: Viewport = useMemo(() => {
    const xs = points.map((p) => p.x);
    const xMin = xs.length ? Math.min(...xs) : -5;
    const xMax = xs.length ? Math.max(...xs) : 5;
    return { xMin: xMin - 0.5, xMax: xMax + 0.5, yMin: -0.2, yMax: 1.2 };
  }, [points]);

  const chart = useMemo(() => {
    const width = 840;
    const height = 360;

    const xs = points.map((p) => p.x);
    const xMin = xs.length ? Math.min(...xs) : -5;
    const xMax = xs.length ? Math.max(...xs) : 5;

    const xGrid = Array.from({ length: 80 }, (_, i) =>
      xMin + ((xMax - xMin) * i) / 79,
    );

    function act(z: number) {
      if (activation === "sigmoid") return 1 / (1 + Math.exp(-z));
      if (activation === "tanh") return (Math.tanh(z) + 1) / 2;
      return clamp01(z);
    }

    const path = xGrid
      .map((xVal, i) => {
        const p = act(w * xVal + b);
        const s = worldToSvg(xVal, p, vp, width, height);
        return `${i === 0 ? "M" : "L"} ${s.x.toFixed(2)} ${s.y.toFixed(2)}`;
      })
      .join(" ");

    const threshold = (() => {
      // z=0 boundary: w*x + b = 0
      if (Math.abs(w) < 1e-9) return null;
      const x0 = -b / w;
      const a = worldToSvg(x0, vp.yMin, vp, width, height);
      const c = worldToSvg(x0, vp.yMax, vp, width, height);
      return { x0, a, c };
    })();

    const dots = points.map((p, idx) => {
      const yJitter = p.y === 1 ? 1 + 0.05 * Math.sin(idx) : 0 - 0.05 * Math.sin(idx);
      const s = worldToSvg(p.x, yJitter, vp, width, height);
      return { idx, x: s.x, y: s.y, label: p.y };
    });

    return { width, height, path, dots, threshold };
  }, [points, vp, w, b, activation]);

  const activationExplain = useMemo(() => {
    if (activation === "sigmoid") {
      return {
        title: "Sigmoid (標準 Logistic Regression)",
        body: "輸出在 (0,1)，可以直接當成 P(y=1|x)。BCE + sigmoid 的梯度很乾淨：∂L/∂z ≈ (p - y)。",
      };
    }
    if (activation === "tanh") {
      return {
        title: "Tanh (先壓到 -1~1，再映射到 0~1)",
        body: "我們用 p=(tanh(z)+1)/2 來當機率。尾端更容易飽和（梯度變小），所以有時會更慢。",
      };
    }
    return {
      title: "Linear (把 z 直接 clamp 到 0~1)",
      body: "這不是標準 logistic regression。它在中間區域梯度像線性，但超出 0~1 會被 clamp，梯度會變 0，學習會卡住。",
    };
  }, [activation]);

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-8 py-10">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          Logistic Regression (1D)
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          模型：<span className="font-mono">p = act(w·x + b)</span>，用 BCE
          loss 來訓練二元分類。
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950 lg:col-span-2">
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Data + Probability Curve
            </h2>
            <div className="font-mono text-xs text-zinc-500 dark:text-zinc-400">
              epoch: {epoch}
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white dark:border-zinc-800 dark:bg-black">
            <svg
              viewBox={`0 0 ${chart.width} ${chart.height}`}
              className="h-auto w-full"
              role="img"
              aria-label="logistic plot"
            >
              <rect
                x={0}
                y={0}
                width={chart.width}
                height={chart.height}
                fill="transparent"
              />

              {chart.threshold ? (
                <line
                  x1={chart.threshold.a.x}
                  y1={chart.threshold.a.y}
                  x2={chart.threshold.c.x}
                  y2={chart.threshold.c.y}
                  stroke="currentColor"
                  className="text-zinc-200 dark:text-zinc-800"
                  strokeWidth={2}
                  strokeDasharray="6 6"
                />
              ) : null}

              <path
                d={chart.path}
                fill="none"
                stroke="currentColor"
                className="text-emerald-600 dark:text-emerald-400"
                strokeWidth={4}
                strokeLinejoin="round"
                strokeLinecap="round"
              />

              {chart.dots.map((d) => (
                <circle
                  key={d.idx}
                  cx={d.x}
                  cy={d.y}
                  r={6}
                  fill="currentColor"
                  className={
                    d.label === 1
                      ? "text-sky-600 dark:text-sky-400"
                      : "text-rose-600 dark:text-rose-400"
                  }
                />
              ))}
            </svg>
          </div>

          <div className="mt-3 text-xs text-zinc-600 dark:text-zinc-400">
            <span className="font-medium">點</span>：y=1（藍）/ y=0（紅）
            ，<span className="font-medium">曲線</span>：模型輸出的 p(x)。
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
          <div className="mb-4">
            <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              Controls
            </h2>
            <div className="mt-1 text-xs text-zinc-500 dark:text-zinc-400">
              {isLoadingPoints ? "loading dataset..." : `n=${points.length}`}
            </div>
          </div>

          <div className="grid gap-3">
            <label className="grid gap-1 text-xs text-zinc-600 dark:text-zinc-400">
              Activation
              <select
                value={activation}
                onChange={(e) => {
                  setActivation(e.target.value as "sigmoid" | "tanh" | "linear");
                  setEpoch(0);
                }}
                className="h-10 rounded-lg border border-zinc-200 bg-white px-3 text-sm text-zinc-900 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50"
                disabled={isTraining}
              >
                <option value="sigmoid">Sigmoid</option>
                <option value="tanh">Tanh</option>
                <option value="linear">Linear (clamp)</option>
              </select>
            </label>

            <div className="rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
              <div className="font-medium text-zinc-800 dark:text-zinc-200">
                {activationExplain.title}
              </div>
              <div className="mt-1 leading-relaxed">{activationExplain.body}</div>
            </div>

            <Slider
              label="w (slope)"
              value={w}
              min={-5}
              max={5}
              step={0.05}
              onChange={setW}
              disabled={isTraining}
            />
            <Slider
              label="b (bias)"
              value={b}
              min={-5}
              max={5}
              step={0.05}
              onChange={setB}
              disabled={isTraining}
            />

            <div className="grid gap-2">
              <button
                type="button"
                onClick={runTrainStep}
                disabled={isTraining || isLoadingPoints || points.length === 0}
                className="h-10 rounded-full border border-zinc-200 bg-white px-4 text-sm font-medium text-zinc-900 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50"
              >
                Step 1 Epoch
              </button>
              <button
                type="button"
                onClick={autoTrain}
                disabled={isTraining || isLoadingPoints || points.length === 0}
                className="h-10 rounded-full bg-foreground px-4 text-sm font-medium text-background disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isTraining ? "Training..." : "Auto Train"}
              </button>
            </div>

            <div className="grid gap-2 rounded-xl border border-zinc-200 bg-zinc-50 p-3 text-xs text-zinc-700 dark:border-zinc-800 dark:bg-zinc-900/40 dark:text-zinc-300">
              <div className="flex items-center justify-between">
                <div className="text-[11px] text-zinc-600 dark:text-zinc-400">BCE loss</div>
                <div className="font-mono text-sm text-zinc-900 dark:text-zinc-50">
                  {Number.isFinite(loss) ? loss.toFixed(4) : "-"}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <Metric label="∂L/∂w" value={gradW} />
                <Metric label="∂L/∂b" value={gradB} />
              </div>
            </div>

            <div className="text-[11px] text-zinc-500 dark:text-zinc-400">
              註：這裡用「全資料」做一步更新（full batch）。之後如果你想
              也可以擴充成 SGD / mini-batch。
            </div>
          </div>
        </div>
      </div>

      <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
        <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
          What is “activation” here?
        </h2>
        <div className="mt-2 grid gap-2 text-sm text-zinc-600 dark:text-zinc-400">
          <div>
            Logistic regression 的核心是把線性輸出 <span className="font-mono">z=w·x+b</span>
            變成機率 <span className="font-mono">p</span>。
          </div>
          <div>
            所以 activation 不是「神經網路層數」那種複雜度，而是這一步：
            <span className="font-mono">p = act(z)</span>。
          </div>
          <div>
            真正標準的是 sigmoid；tanh/linear 只是用來對照「梯度飽和」與
            「clamp 導致學不動」的差異。
          </div>
        </div>
      </div>
    </main>
  );
}

function Slider(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <label className="grid gap-1 text-xs text-zinc-600 dark:text-zinc-400">
      {props.label}
      <div className="flex items-center gap-3">
        <input
          type="range"
          min={props.min}
          max={props.max}
          step={props.step}
          value={props.value}
          onChange={(e) => props.onChange(Number(e.target.value))}
          className="w-full"
          disabled={props.disabled}
        />
        <div className="w-20 rounded-lg border border-zinc-200 bg-white px-2 py-1 font-mono text-xs text-zinc-900 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-50">
          {props.value.toFixed(2)}
        </div>
      </div>
    </label>
  );
}

function Metric(props: { label: string; value: number }) {
  return (
    <div className="rounded-lg border border-zinc-200 bg-white p-2 dark:border-zinc-800 dark:bg-zinc-950">
      <div className="text-[11px] text-zinc-600 dark:text-zinc-400">{props.label}</div>
      <div className="font-mono text-xs text-zinc-900 dark:text-zinc-50">
        {Number.isFinite(props.value) ? props.value.toFixed(4) : "-"}
      </div>
    </div>
  );
}
