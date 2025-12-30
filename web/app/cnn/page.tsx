"use client";

import { useEffect, useRef, useState } from "react";

type PredictResponse = {
  predicted_digit: number;
  probabilities: number[];
  logits: number[];
  conv1_maps?: number[][][]; // (16, H, W)
  conv2_maps?: number[][][]; // (32, H, W)
};

export default function CNNPage() {
  const API_BASE =
    process.env.NEXT_PUBLIC_API_BASE?.trim() || "http://localhost:8000";

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isDrawing, setIsDrawing] = useState(false);
  const [prediction, setPrediction] = useState<PredictResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [showFeatureMaps, setShowFeatureMaps] = useState(false);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Initialize with white background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }, []);

  function startDrawing(e: React.MouseEvent<HTMLCanvasElement>) {
    setIsDrawing(true);
    draw(e);
  }

  function stopDrawing() {
    setIsDrawing(false);
  }

  function draw(e: React.MouseEvent<HTMLCanvasElement>) {
    if (!isDrawing && e.type !== "mousedown") return;

    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;

    const x = (e.clientX - rect.left) * scaleX;
    const y = (e.clientY - rect.top) * scaleY;

    ctx.fillStyle = "#000000";
    ctx.beginPath();
    ctx.arc(x, y, 12, 0, Math.PI * 2);
    ctx.fill();
  }

  function clearCanvas() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    setPrediction(null);
  }

  async function predictDigit() {
    const canvas = canvasRef.current;
    if (!canvas) return;

    setIsLoading(true);
    try {
      const dataURL = canvas.toDataURL("image/png");

      const res = await fetch(`${API_BASE}/predict/mnist`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: dataURL,
          return_feature_maps: showFeatureMaps,
        }),
      });

      if (!res.ok) throw new Error(`Prediction failed: ${res.status}`);

      const data = (await res.json()) as PredictResponse;
      setPrediction(data);
    } catch (err) {
      console.error(err);
      alert("Prediction failed. Make sure the backend is running.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="mx-auto flex w-full max-w-7xl flex-col gap-8 px-8 py-10">
      <div className="flex flex-col gap-2">
        <h1 className="text-2xl font-semibold tracking-tight">
          CNN: MNIST Digit Recognition
        </h1>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          手寫一個數字（0-9），看 CNN 如何辨識並提取特徵。
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Drawing Canvas */}
        <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
          <div className="mb-3 flex items-baseline justify-between">
            <h2 className="text-sm font-medium text-zinc-700 dark:text-zinc-300">
              手寫輸入區
            </h2>
            <button
              type="button"
              onClick={clearCanvas}
              className="text-xs text-zinc-500 hover:text-zinc-900 dark:text-zinc-400 dark:hover:text-zinc-50"
            >
              清除
            </button>
          </div>

          <div className="flex flex-col items-center gap-4">
            <canvas
              ref={canvasRef}
              width={280}
              height={280}
              onMouseDown={startDrawing}
              onMouseMove={draw}
              onMouseUp={stopDrawing}
              onMouseLeave={stopDrawing}
              className="cursor-crosshair rounded-xl border-2 border-zinc-300 bg-white dark:border-zinc-700"
              style={{ touchAction: "none" }}
            />

            <div className="flex w-full flex-col gap-2">
              <label className="flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
                <input
                  type="checkbox"
                  checked={showFeatureMaps}
                  onChange={(e) => setShowFeatureMaps(e.target.checked)}
                  className="rounded"
                />
                顯示 Feature Maps（第一層卷積）
              </label>

              <button
                type="button"
                onClick={predictDigit}
                disabled={isLoading}
                className="h-12 rounded-full bg-foreground px-6 text-sm font-medium text-background disabled:cursor-not-allowed disabled:opacity-60"
              >
                {isLoading ? "辨識中..." : "辨識數字"}
              </button>
            </div>
          </div>
        </div>

        {/* Prediction Results */}
        <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
          <h2 className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
            預測結果
          </h2>

          {prediction ? (
            <div className="flex flex-col gap-4">
              <div className="flex items-center justify-between rounded-xl border border-emerald-200 bg-emerald-50 p-4 dark:border-emerald-800 dark:bg-emerald-950/30">
                <div className="text-sm text-emerald-800 dark:text-emerald-200">
                  預測數字
                </div>
                <div className="text-4xl font-bold text-emerald-600 dark:text-emerald-400">
                  {prediction.predicted_digit}
                </div>
              </div>

              <div className="flex flex-col gap-2">
                <div className="text-xs text-zinc-600 dark:text-zinc-400">
                  各類別機率分佈
                </div>
                {prediction.probabilities.map((prob, idx) => {
                  const isMax = idx === prediction.predicted_digit;
                  return (
                    <div key={idx} className="flex items-center gap-2">
                      <div className="w-8 text-right font-mono text-xs text-zinc-600 dark:text-zinc-400">
                        {idx}
                      </div>
                      <div className="relative h-6 flex-1 overflow-hidden rounded-lg bg-zinc-100 dark:bg-zinc-900">
                        <div
                          className={`h-full transition-all ${
                            isMax
                              ? "bg-emerald-500 dark:bg-emerald-400"
                              : "bg-zinc-300 dark:bg-zinc-700"
                          }`}
                          style={{ width: `${prob * 100}%` }}
                        />
                        <div className="absolute inset-0 flex items-center justify-end px-2 font-mono text-xs text-zinc-900 dark:text-zinc-50">
                          {(prob * 100).toFixed(1)}%
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="flex h-64 items-center justify-center text-sm text-zinc-500 dark:text-zinc-400">
              畫一個數字並點擊「辨識數字」
            </div>
          )}
        </div>
      </div>

      {/* CNN Architecture Explanation */}
      <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
        <h2 className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
          CNN 架構：這個模型怎麼運作？
        </h2>

        <div className="flex flex-col gap-4">
          <div className="overflow-x-auto">
            <div className="flex items-center gap-2 text-xs">
              <LayerBox label="輸入" detail="1×28×28" />
              <Arrow />
              <LayerBox label="Conv1" detail="16 filters" color="emerald" />
              <Arrow />
              <LayerBox label="ReLU" detail="activation" />
              <Arrow />
              <LayerBox label="MaxPool" detail="2×2" />
              <Arrow />
              <LayerBox label="Conv2" detail="32 filters" color="sky" />
              <Arrow />
              <LayerBox label="ReLU" detail="activation" />
              <Arrow />
              <LayerBox label="MaxPool" detail="2×2" />
              <Arrow />
              <LayerBox label="FC" detail="128→10" color="violet" />
            </div>
          </div>

          <div className="grid gap-3 text-sm text-zinc-600 dark:text-zinc-400">
            <div>
              <span className="font-medium text-zinc-800 dark:text-zinc-200">
                卷積層（Conv）
              </span>
              ：用小窗格掃過圖片，找出「邊緣」、「角落」、「圓弧」等基本特徵。
            </div>
            <div>
              <span className="font-medium text-zinc-800 dark:text-zinc-200">
                ReLU
              </span>
              ：去掉負值，讓模型只保留「有反應」的部分，增加非線性。
            </div>
            <div>
              <span className="font-medium text-zinc-800 dark:text-zinc-200">
                MaxPool
              </span>
              ：把圖片縮小一半，保留最強的訊號，減少計算量。
            </div>
            <div>
              <span className="font-medium text-zinc-800 dark:text-zinc-200">
                全連接層（FC）
              </span>
              ：把所有特徵匯總，最後輸出 10 個分數（0-9 的機率）。
            </div>
          </div>
        </div>
      </div>

      {/* Feature Maps Visualization */}
      {prediction?.conv1_maps && (
        <div className="rounded-2xl border border-zinc-200 bg-white p-5 dark:border-zinc-800 dark:bg-zinc-950">
          <h2 className="mb-3 text-sm font-medium text-zinc-700 dark:text-zinc-300">
            第一層卷積特徵圖（Conv1 - 16 filters）
          </h2>
          <p className="mb-4 text-xs text-zinc-600 dark:text-zinc-400">
            每個小圖代表一個 filter「看到」的特徵：某些偵測水平線、某些偵測垂直線、某些偵測角落。
          </p>

          <div className="grid grid-cols-8 gap-2">
            {prediction.conv1_maps.map((fmap, idx) => (
              <FeatureMapCanvas key={idx} data={fmap} label={`F${idx}`} />
            ))}
          </div>
        </div>
      )}
    </main>
  );
}

function LayerBox(props: {
  label: string;
  detail: string;
  color?: "emerald" | "sky" | "violet";
}) {
  const colorClass = props.color
    ? props.color === "emerald"
      ? "border-emerald-200 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/30"
      : props.color === "sky"
        ? "border-sky-200 bg-sky-50 dark:border-sky-800 dark:bg-sky-950/30"
        : "border-violet-200 bg-violet-50 dark:border-violet-800 dark:bg-violet-950/30"
    : "border-zinc-200 bg-zinc-50 dark:border-zinc-800 dark:bg-zinc-900/40";

  return (
    <div
      className={`flex min-w-[80px] flex-col items-center gap-1 rounded-lg border p-2 ${colorClass}`}
    >
      <div className="font-medium text-zinc-900 dark:text-zinc-50">
        {props.label}
      </div>
      <div className="text-[10px] text-zinc-600 dark:text-zinc-400">
        {props.detail}
      </div>
    </div>
  );
}

function Arrow() {
  return (
    <svg
      width="20"
      height="20"
      viewBox="0 0 20 20"
      fill="none"
      className="text-zinc-400 dark:text-zinc-600"
    >
      <path
        d="M4 10h12m0 0l-4-4m4 4l-4 4"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function FeatureMapCanvas(props: { data: number[][]; label: string }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const h = props.data.length;
    const w = props.data[0]?.length || 0;
    if (h === 0 || w === 0) return;

    canvas.width = w;
    canvas.height = h;

    // Find min/max for normalization
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const val = props.data[i][j];
        min = Math.min(min, val);
        max = Math.max(max, val);
      }
    }

    const range = max - min || 1;

    const imageData = ctx.createImageData(w, h);
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const val = (props.data[i][j] - min) / range;
        const gray = Math.floor(val * 255);
        const idx = (i * w + j) * 4;
        imageData.data[idx] = gray;
        imageData.data[idx + 1] = gray;
        imageData.data[idx + 2] = gray;
        imageData.data[idx + 3] = 255;
      }
    }

    ctx.putImageData(imageData, 0, 0);
  }, [props.data]);

  return (
    <div className="flex flex-col items-center gap-1">
      <canvas
        ref={canvasRef}
        className="h-auto w-full rounded border border-zinc-200 dark:border-zinc-800"
        style={{ imageRendering: "pixelated" }}
      />
      <div className="text-[10px] text-zinc-500 dark:text-zinc-400">
        {props.label}
      </div>
    </div>
  );
}
