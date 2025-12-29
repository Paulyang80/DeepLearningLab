# Logistic Regression 視覺化專案說明（Next.js + FastAPI + NumPy）

本單元的目標：把 **Logistic Regression（二元分類）** 變成「看得到、摸得到」的互動視覺化。

你會在網頁上調整參數 $w,b$ 與 activation，立刻看到：

- 分類機率曲線 $p(x)$ 如何改變
- BCE loss（交叉熵）如何變大/變小
- 梯度 $\frac{\partial L}{\partial w}, \frac{\partial L}{\partial b}$ 指向哪裡

本專案對應頁面：`/logistic`

---

## 1) Logistic Regression 在做什麼？（一張圖就懂）

### 1.1 線性輸出（logit）

先算一個線性值：

$$z = wx + b$$

- $w$：斜率（決定曲線「翻轉」位置與陡峭程度）
- $b$：偏置（把整個邊界左右平移）

### 1.2 變成機率（activation）

再把 $z$ 經過 activation 變成 $p \in (0,1)$：

$$p = \text{act}(z)$$

在這個視覺化裡，你可以切換 activation：

- **Sigmoid（標準）**：$p=\sigma(z)=\frac{1}{1+e^{-z}}$
- **Tanh（對照用）**：我們用 $p=\frac{\tanh(z)+1}{2}$ 把範圍映射到 0~1
- **Linear clamp（對照用）**：$p=\text{clip}(z,0,1)$（不是標準 logistic regression）

> 重點：Logistic regression 的「logistic」其實就來自 sigmoid 這個把線性值壓成機率的函數。

---

## 2) Loss：為什麼用 BCE（Binary Cross Entropy）？

對二元標籤 $y\in\{0,1\}$，BCE 定義：

$$L = -\frac{1}{n}\sum_{i=1}^{n}\Big(y_i\log(p_i) + (1-y_i)\log(1-p_i)\Big)$$

直覺：

- 若 $y=1$，希望 $p$ 越接近 1 越好（$\log(p)$ 越大）
- 若 $y=0$，希望 $p$ 越接近 0 越好（$\log(1-p)$ 越大）

BCE 會「重罰」自信但錯的預測（例如 $y=1$ 卻給 $p\approx0$）。

---

## 3) 梯度直覺：為什麼 sigmoid + BCE 好用？

令 $z_i=wx_i+b$、$p_i=\sigma(z_i)$。

對 sigmoid + BCE，有一個非常漂亮的結果：

$$\frac{\partial L}{\partial z_i} \approx (p_i - y_i)$$

因此：

$$\frac{\partial L}{\partial w} = \frac{1}{n}\sum_i (p_i-y_i)x_i,\quad
\frac{\partial L}{\partial b} = \frac{1}{n}\sum_i (p_i-y_i)$$

直覺：

- 若某點 $y=1$ 但 $p$ 太小，$p-y<0$，更新會把 $z$ 拉大（提高機率）
- 若某點 $y=0$ 但 $p$ 太大，$p-y>0$，更新會把 $z$ 拉小（降低機率）

這就是你在 `/logistic` 看到梯度顯示時「它在推你往哪走」的數學原因。

---

## 4) 本專案的視覺化怎麼讀？

頁面：`/logistic`

- **點（藍/紅）**：資料標籤 $y$
  - 藍：$y=1$
  - 紅：$y=0$
- **綠色曲線**：模型輸出的機率 $p(x)=\text{act}(wx+b)$
- **虛線**（若有）：決策邊界 $z=0$，即 $wx+b=0$ 的位置

你可以做的操作：

- 手動調 $w,b$ 看曲線怎麼動
- 切換 activation，看「同一個 $z$」映射成機率時的差異
- 按 `Step 1 Epoch` 做一次 full-batch 更新
- 按 `Auto Train` 連續做多步更新

---

## 5) FastAPI / NumPy 後端 API 對照

後端檔案：`backend/main.py`

### 5.1 產生資料集

- `POST /dataset/logistic_1d`
- 會用 `true_w,true_b` 生成真實機率 $p=\sigma(true\_w\,x+true\_b)$，再抽樣得到 $y\in\{0,1\}$。

範例：

```bash
curl -sS -X POST http://localhost:8000/dataset/logistic_1d \
  -H 'Content-Type: application/json' \
  -d '{"n":60,"seed":7,"x_min":-5,"x_max":5,"true_w":1.2,"true_b":-0.3}'
```

### 5.2 解釋（forward + loss + gradient）

- `POST /explain/logistic_1d`
- 回傳：
  - `loss`：BCE
  - `grad_w, grad_b`
  - `probs`：每個點的 $p_i$
  - `errors`：$p_i - y_i$（常見的直覺誤差）

範例：

```bash
curl -sS -X POST http://localhost:8000/explain/logistic_1d \
  -H 'Content-Type: application/json' \
  -d '{"w":1,"b":0,"activation":"sigmoid","points":[{"x":-1,"y":0},{"x":1,"y":1}]}'
```

### 5.3 訓練一步（Gradient Descent Step）

- `POST /train/logistic_1d/step`
- 輸入目前 `w,b` 與 `learning_rate`，回傳更新後的 `w,b` + 同一批資料的 loss/grad/probs。

---

## 6) activation 的教學重點（你應該觀察什麼）

### 6.1 Sigmoid（標準答案）

- 機率輸出自然且連續
- 搭配 BCE，梯度訊號乾淨，最適合教 logistic regression

### 6.2 Tanh（對照：飽和與梯度變小）

- 我們把 tanh 映射到 0~1 才能當機率
- 在 $|z|$ 很大時容易飽和，$\frac{dp}{dz}$ 變很小，更新可能變慢

### 6.3 Linear clamp（對照：clamp 會讓你「學不動」）

- 這不是標準 logistic regression
- 超出 0~1 後會被 clamp，導致梯度為 0（你會看到訓練卡住）

這三個放在同個 UI 裡的目的：讓你在「同一組資料」上感覺到 activation 對訓練動力學的影響。

---

## 7) 如何在本機跑起來（macOS）

### 7.1 啟動後端（FastAPI）

> 你的環境可能沒有 `python` 這個指令，通常是 `python3`。

```bash
cd /Users/paulyang/Projects/DeepLearningLab
python3 -m uvicorn backend.main:app --reload --port 8000
```

確認：

```bash
curl -sS http://localhost:8000/health
```

### 7.2 啟動前端（Next.js）

```bash
cd /Users/paulyang/Projects/DeepLearningLab/web
npm run dev
```

打開：

- http://localhost:3000/logistic

---

## 8) 常見卡關（快速排查）

- **看不到曲線/資料**：先確認 `curl http://localhost:8000/health` 是否回 `{"ok": true}`
- **CORS**：後端已用 regex 放行 localhost/127.0.0.1 任意 port；若你改了 domain 再調整
- **loss 變 NaN**：BCE 需要 $p$ 避免 0 或 1，後端有做 clip；若你自己改動記得保留

---

## 9) 下一步建議

如果你想把 logistic 頁也做得跟 linear 頁一樣完整，可以加：

- `loss vs epoch` 圖
- Full batch / SGD / Mini-batch 模式與 epoch/iteration 說明
- 決策邊界動畫（$x_0=-b/w$ 的移動）