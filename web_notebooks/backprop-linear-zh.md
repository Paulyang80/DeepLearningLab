# 線性回歸的 Backprop（從零推到會）

目標：把你現在看到的「結果」變成腦中有直覺的「過程」。

---

## 1) 模型（Model）

我們的線性模型是：

\[
\hat{y} = wx + b
\]

- **資料（data）**：$x, y$（每個資料點都給定）
- **模型參數（parameters）**：$w, b$（我們要學的）
- **模型輸出（prediction）**：$\hat{y}$（由 $x$ 和參數算出來）

對於第 $i$ 個資料點：

\[
\hat{y}_i = wx_i + b
\]

---

## 2) 損失函數（Loss, MSE）

我們用均方誤差（Mean Squared Error）：

\[
L(w,b) = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)^2
\]

把 $\hat{y}_i = wx_i + b$ 代回去：

\[
L(w,b) = \frac{1}{n}\sum_{i=1}^{n}(wx_i + b - y_i)^2
\]

定義「誤差」：

\[
e_i = \hat{y}_i - y_i = (wx_i + b - y_i)
\]

那麼：

\[
L(w,b) = \frac{1}{n}\sum_{i=1}^{n} e_i^2
\]

直覺：
- $e_i$ 是「你預測錯多少」
- $e_i^2$ 讓正負號消失，並且「錯得越多懲罰越大」

---

## 3) 梯度推導（逐步，不省略）

我們要：

\[
\frac{\partial L}{\partial w},\quad \frac{\partial L}{\partial b}
\]

### 3.1 先推 \(\partial L / \partial w\)

從定義開始：

\[
L(w,b)=\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2
\]

先把求導拆成「外層平均」+「內層每一項」：

\[
\frac{\partial L}{\partial w}=
\frac{\partial}{\partial w}\left(\frac{1}{n}\sum_{i=1}^{n}(wx_i+b-y_i)^2\right)
\]

常數 $1/n$ 可以提出來：

\[
\frac{\partial L}{\partial w}=
\frac{1}{n}\sum_{i=1}^{n}\frac{\partial}{\partial w}(wx_i+b-y_i)^2
\]

現在看單一項：

\[
\frac{\partial}{\partial w}(wx_i+b-y_i)^2
\]

令 $u_i = wx_i + b - y_i$，那就是 $u_i^2$。

- 外層：$\frac{d}{du}(u^2)=2u$
- 內層：$\frac{\partial u_i}{\partial w}=\frac{\partial}{\partial w}(wx_i+b-y_i)=x_i$

用鏈式法則（chain rule）：

\[
\frac{\partial}{\partial w}(u_i^2)=2u_i\cdot\frac{\partial u_i}{\partial w}=2(wx_i+b-y_i)\cdot x_i
\]

代回總和：

\[
\frac{\partial L}{\partial w}=
\frac{1}{n}\sum_{i=1}^{n} 2(wx_i+b-y_i)x_i
\]

整理：

\[
\boxed{\frac{\partial L}{\partial w}=
\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)x_i}
\]

**每一項的意義：**
- $(\hat{y}_i-y_i)$：來自資料點的「誤差」
- $x_i$：來自資料點的輸入，告訴你「斜率 w 改一點會對這個點造成多大影響」
- 乘起來：誤差越大、且 $|x_i|$ 越大，對 $w$ 的推動越大

### 3.2 再推 \(\partial L / \partial b\)

同樣從：

\[
\frac{\partial L}{\partial b}=
\frac{1}{n}\sum_{i=1}^{n}\frac{\partial}{\partial b}(wx_i+b-y_i)^2
\]

對單一項：令 $u_i=wx_i+b-y_i$。

- 外層：$\frac{d}{du}(u^2)=2u$
- 內層：$\frac{\partial u_i}{\partial b}=\frac{\partial}{\partial b}(wx_i+b-y_i)=1$

鏈式法則：

\[
\frac{\partial}{\partial b}(u_i^2)=2u_i\cdot 1 = 2(wx_i+b-y_i)
\]

代回：

\[
\boxed{\frac{\partial L}{\partial b}=
\frac{2}{n}\sum_{i=1}^{n}(\hat{y}_i-y_i)}
\]

**每一項的意義：**
- $b$ 是整條線上下平移，所以每個點對 $b$ 的敏感度都是 1
- 因此只需要把所有誤差加總（再平均）就能決定 $b$ 往哪裡走

---

## 4) 用「一行公式 ↔ 一段 NumPy」對照

以下程式碼刻意拆成：
- forward pass
- loss computation
- gradient computation
- parameter update

假設：
- `x.shape == (n,)`
- `y.shape == (n,)`
- `w,b` 是純 float

```python
import numpy as np

# data: x, y
# params: w, b

# --- forward pass ---
# 公式：y_hat = w x + b
y_hat = w * x + b

# --- loss (MSE) ---
# 公式：e = y_hat - y
err = y_hat - y

# 公式：L = (1/n) Σ e^2
loss = np.mean(err ** 2)

# --- gradients ---
# 公式：∂L/∂w = (2/n) Σ e * x
# 等價：2 * mean(e * x)
dw = 2.0 * np.mean(err * x)

# 公式：∂L/∂b = (2/n) Σ e
# 等價：2 * mean(e)
db = 2.0 * np.mean(err)

# --- update (Gradient Descent) ---
# 公式：w <- w - lr * dw
#      b <- b - lr * db
w = w - lr * dw
b = b - lr * db
```

你現在在 UI 裡看到的：
- 每個點的 `err_i`（正負用顏色）
- 整體的 `dw, db`
- 以及每一步的 `w_t → w_{t+1}`, `b_t → b_{t+1}`
就是把上面這段拆給你看。

---

## 5) 用人話回答你最在意的三個問題

### (1) Backpropagation 在這個例子裡到底在幹嘛？
它在做「責任分攤」：
- 先算出每個點錯多少（error）
- 再算出這個錯誤要怪給參數多少（gradient）
- 最後告訴你：
  - **w** 應該往哪邊改、改多少
  - **b** 應該往哪邊改、改多少

在這個一層線性模型中，backprop 的核心就是鏈式法則：
- $L \rightarrow e \rightarrow (w,b)$

### (2) 為什麼不用 backprop，我就只能用滑桿亂試？
因為滑桿只是在「猜方向」。
你可以看到 loss 變大或變小，但你不知道：
- 到底是因為哪幾個點的誤差在拉你
- 是該先改 w 還是先改 b
- 改多少才合理

gradient（梯度）把這些都量化成「方向」和「大小」。

### (3) PyTorch 的 loss.backward() 在幫我做哪幾件事？
它幫你自動做：
1) forward：算出 $\hat{y}$
2) loss：算出 $L$
3) backward：用鏈式法則把 $\partial L/\partial w, \partial L/\partial b$ 算出來

你現在在 NumPy 手動做的 `dw, db`，未來就是 `w.grad`, `b.grad`。

---

## 6) 自然延伸（不超載）

1) **把 x 變成多維（多元線性回歸）**：$\hat{y} = \mathbf{w}^T\mathbf{x} + b$。
   - 合理原因：梯度的形式會變成向量，但邏輯完全一樣（誤差 × 輸入）。

2) **改用 MAE 或 Huber loss**
   - 合理原因：你會看到「損失的形狀」改變，梯度的行為也會改變（對 outlier 的敏感度不同）。

3) **畫出 loss surface（w-b 平面）與下降軌跡**
   - 合理原因：你會直接看到「為什麼沿著 -grad 會往谷底走」，把方向感建立起來。
