# 線性模型視覺化專案說明（Next.js + TypeScript + Tailwind）

本專案的目標是用網頁把簡單線性模型視覺化：

$$y = wx + b$$

你可以在頁面上調整參數 $w$（斜率）與 $b$（截距），立刻看到直線如何改變。
這種「看得到、摸得到」的互動式視覺化，能幫你把抽象的 ML/DL 概念（參數、模型輸出、損失、梯度）變成直覺。

這份文件面向的讀者：

- 有 Python 後端經驗（FastAPI / Flask）
- 上過 Angular（TypeScript）
- 對 JavaScript / React / Next.js 生態系還不熟
- 正在學 DL，希望從視覺化建立直覺

---

## 1) 核心名詞與概念（非常重要）

這一節的目標是：**把每個名詞放到正確的分類：語言、執行環境、函式庫、框架**。
很多初學者卡住的原因不是寫不出程式，而是「名詞混在一起」，例如把 Node.js 當成語言、把 React 當成後端。

### 1.1 一張總覽表：它到底是什麼？主要做哪邊？

| 名稱 | 它是什麼（language/runtime/library/framework） | 主要在做哪邊 | 用一句話記住它 |
|---|---|---|---|
| HTML | 標記語言（markup language） | 前端 | 定義網頁結構（內容是什麼） |
| CSS | 樣式表語言（style sheet language） | 前端 | 定義網頁外觀（長什麼樣子） |
| JavaScript (JS) | 程式語言（programming language） | 前端＋後端（透過 Node.js） | 讓網頁「動起來」的語言 |
| TypeScript (TS) | 程式語言（JS 的超集） | 開發時（最後編譯成 JS） | 有型別的 JS，適合大型專案 |
| Node.js | 執行環境（runtime） | 後端／工具鏈 | 讓 JS 能在瀏覽器外執行 |
| React | 函式庫（library） | 前端為主 | 用元件與狀態描述 UI |
| Next.js | 框架（framework，基於 React） | 前端＋後端 | React 的全端框架（頁面＋API） |
| Tailwind CSS | CSS 工具型框架（utility-first CSS framework） | 前端 | 用 class 組出樣式，少寫自訂 CSS |
| Angular | 框架（framework） | 前端 | 大而全的前端框架（路由、DI 等都內建） |

> 注意：HTML/CSS 常被口語稱為「語言」，但不是「程式語言」。它們主要負責結構與樣式，不負責計算與流程控制。

---

### 1.2 HTML / CSS / JavaScript：三者分工（重要）

你可以用「房子」比喻：

- **HTML**：房子的骨架與隔間（哪裡是標題、哪裡是按鈕、哪裡是圖表區）
- **CSS**：房子的裝潢（顏色、字體、排版、間距）
- **JavaScript**：房子的電路與互動（按按鈕會發生什麼、滑桿改變就更新圖形）

ASCII 示意：

```
[HTML] 內容與結構
   |
   v
[CSS]  外觀與排版
   |
   v
[JS]   行為與互動
```

對應到線性模型視覺化：

- HTML：放「w 滑桿」、「b 滑桿」、「顯示方程式」、「畫布/圖表區」
- CSS / Tailwind：讓滑桿與圖表排版整齊、字體清晰
- JS/TS（React）：當 w/b 變動時，重新計算直線並更新畫面

---

### 1.3 什麼是 JavaScript？

**JavaScript（JS）是程式語言。**

- 在瀏覽器中：處理使用者互動、更新畫面、呼叫 API
- 在 Node.js 中：可以寫伺服器程式、做建置工具、跑開發伺服器

對 Python 使用者的對照：

- Python 是你熟悉的語言；JS 也是語言，只是主要應用場域從一開始就偏向 Web
- 兩者都能做後端，但 Python 在科學計算、ML/DL 生態系更強

---

### 1.4 什麼是 TypeScript？和 JavaScript 的關係是什麼？

**TypeScript（TS）是 JavaScript 的超集（superset）。**
意思是：

- ✅ 所有合法的 JavaScript 程式碼，都是合法的 TypeScript
- ✅ TypeScript 額外加入「型別系統」與一些語言層級能力
- ✅ TypeScript 最終會被「編譯（transpile）」成 JavaScript 才能執行

關係圖：

```
你寫： TypeScript (含型別)
   |
   |  tsc / Next.js build（轉換）
   v
輸出： JavaScript (真正被執行)
```

為什麼要用 TS？（用例子理解）

假設你要把滑桿輸入轉成數字：

- 在 JS 裡，字串和數字很容易混在一起，錯了可能要到執行時才發現
- 在 TS 裡，你可以把狀態型別定清楚（例如 `w: number`），工具就能更早提醒你

你上過 Angular（TS）：

- Angular 幾乎「預設 TS」
- Next.js + React 也非常推薦 TS，尤其是要做互動視覺化、狀態較多時

---

### 1.5 什麼是 Node.js？（它不是程式語言）

**Node.js 是 JavaScript 的執行環境（runtime），不是程式語言。**

- JavaScript：你寫的程式「用哪種語言寫」
- Node.js：這段 JS「在哪裡跑」以及「能用哪些系統 API」

用一個非常直覺的對照：

- Python 程式要跑，通常需要 Python interpreter（例如 `python3`）
- JavaScript 程式要在伺服器/本機跑，需要 Node.js

ASCII 示意：

```
JS/TS 程式碼
   |
   v
Node.js Runtime
(檔案、網路、process 等 API)
   |
   v
作業系統 / 伺服器
```

在 Next.js 裡，Node.js 扮演的角色通常是：

- 跑開發伺服器（`npm run dev`）
- 執行建置（`npm run build`）
- 執行 Next.js 的伺服器端程式（例如 API route、Server Components）

---

### 1.6 什麼是 React？

**React 是前端 UI 函式庫（library）。**

它的核心思想：

- 你用「元件（Component）」把畫面拆成一塊一塊
- 元件有「狀態（state）」
- 當狀態改變，畫面會跟著更新（你不用自己手動改 DOM）

用本專案理解 React：

- `w`、`b` 是 state
- state 改變 → 重新計算直線 → 重新渲染圖表

你可以把 React 想成：

- 「把 UI 變成一個由狀態驅動的函數」

簡化圖：

```
state (w, b)
   |
   v
UI = f(state)
```

---

### 1.7 什麼是 Next.js？為什麼它被說是全端框架？

**Next.js 是基於 React 的框架（framework）。**

它被稱為「全端」的主要原因：

- 它同時處理 **前端頁面（React UI）**
- 也能在同一個專案內提供 **後端 API（Route Handlers / API Routes）**
- 還包含「路由、建置、部署、不同渲染策略」等整合能力

一張直覺的對照（非常粗略但好記）：

- React：主要負責 UI（像是「只管 View」）
- Next.js：把 UI + 路由 + 伺服器能力 + 工程化都整合（像是「完整 Web App 框架」）

ASCII 示意：

```
Browser
  |
  v
Next.js App
  |- Pages/UI (React)
  |- Routing
  |- API (BFF)
  |- Build/Deploy tooling
```

#### 什麼是 BFF（Backend For Frontend）？

BFF 可以理解成「為前端量身打造的後端層」，常見用途：

- 把多個後端服務的資料整合成前端需要的格式
- 處理認證、session、cookie
- 隱藏後端服務細節，讓前端呼叫更簡單

在「Next.js + Python DL」架構中：

- Next.js API/BFF：負責前端友善的 API
- Python FastAPI：負責真正的 ML/DL 計算與推論

---

### 1.8 什麼是 Tailwind CSS？

**Tailwind CSS 是「工具型」的 CSS 框架（utility-first）。**

傳統 CSS 可能會寫：

- 先命名 class：`.card { ... }`
- 再把 class 掛到 HTML 上

Tailwind 的思路是：

- 直接在元素上用一堆小工具 class 組合出樣式

例如（概念示意）：

- `p-4`：padding
- `text-lg`：字體大小
- `flex gap-2`：排版

優點（對教學專案尤其有利）：

- 排版速度快
- 不用一直在 CSS 檔案和元件檔案間跳來跳去
- 介面更容易保持一致

---

## 2) 用你熟悉的 Angular / Python 角度對照（補充）

### 2.1 Angular vs React vs Next.js（用框架定位來理解）

| 你熟悉的 | 對應到這邊的概念 | 備註 |
|---|---|---|
| Angular（framework） | Next.js（framework） | 都是比較「大而全」的做法 |
| Angular Component | React Component | 元件化概念相近 |
| Angular Router | Next.js Router（App Router） | 都在處理頁面路由 |
| Angular Service + HttpClient | `fetch` / API client / server actions 等 | 作法不同，但目的類似 |

差異（初學常見感受）：

- Angular：很多東西「框架幫你定好了」
- React：更像「UI 核心 + 自由組合」，而 Next.js 再把周邊補齊

### 2.2 HTML/CSS 在 React/Next.js 裡長什麼樣子？

React 使用的是 **JSX/TSX**（你可以把它想成「把 HTML 樣子的東西放在 JS/TS 裡寫」）：

- 看起來像 HTML
- 但它其實是語法糖，最後會變成 JS 的函數呼叫

對初學者的建議：

- 不要糾結「這到底是不是 HTML」
- 先把它當成「用來描述 UI 結構的語法」就好

### 2.3 如果我要用 Python 當後端：Next.js + FastAPI 最小串接

這一小節的目標是：把「前端互動（滑桿、圖形）」和「Python 計算（DL/ML）」接起來，形成你之後做深度學習 demo 的基本架構。

你可以先用線性模型當範例（因為簡單），之後再把同樣的 API 形狀換成 PyTorch 推論 / 訓練。

#### 2.3.1 架構選擇（你有兩種常見做法）

**做法 A：瀏覽器直接呼叫 FastAPI（最直覺）**

```
Browser (fetch)
  |
  v
FastAPI (Python)
```

- 優點：最少搬運、最好理解
- 缺點：需要處理 CORS；如果有金鑰/機密，不適合放在瀏覽器端

**做法 B：Next.js 當 BFF 轉接到 FastAPI（更像真實產品）**

```
Browser
  |
  v
Next.js Route Handler (Node/BFF)
  |
  v
FastAPI (Python)
```

- 優點：避免 CORS、可以集中管理 token / session、前端呼叫更單純
- 缺點：多一層要維護

入門教學建議：先做 A（快），熟了再加 B（更乾淨）。

#### 2.3.2 FastAPI：最小可用 API（先用線性模型示範）

你可以先做一個「吃 $w,b,x$ 回傳 $y$」的 API，之後把內部計算換成 PyTorch 模型即可。

建立一個 `backend/main.py`（檔名可自訂）概念如下：

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


app = FastAPI(title="DL Visualizer API")


# 做法 A 需要 CORS（只在開發期先這樣開）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LineRequest(BaseModel):
    w: float
    b: float
    x: float


class LineResponse(BaseModel):
    y: float


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/predict/line", response_model=LineResponse)
def predict_line(req: LineRequest):
    return {"y": req.w * req.x + req.b}
```

安裝與啟動（示意）：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install fastapi uvicorn
uvicorn backend.main:app --reload --port 8000
```

> 小提醒：真正在 repo 裡落地時，我會建議把 FastAPI 的依賴寫在 `backend/requirements.txt`，避免跟你的 DL 依賴混在一起；但教學理解上先不用分太細也可以。

#### 2.3.3 Next.js：從前端呼叫 Python API（fetch）

在 Next.js（React）裡，你可以在滑桿變動時呼叫 API。

概念範例（只示範呼叫方式；你可以把它放在 client component 裡）：

```ts
async function fetchY(w: number, b: number, x: number) {
  const res = await fetch("http://localhost:8000/predict/line", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ w, b, x }),
  });

  if (!res.ok) throw new Error(`API error: ${res.status}`);
  const data = (await res.json()) as { y: number };
  return data.y;
}
```

開發時你會同時跑兩個服務：

- Next.js：`npm run dev`（通常 `http://localhost:3000`）
- FastAPI：`uvicorn ... --port 8000`（`http://localhost:8000`）

#### 2.3.4 把「線性模型 API」換成「深度學習推論 API」要注意什麼？

當你開始用 PyTorch 做推論，API 外觀仍然可以長得像這樣：

- `POST /predict`：輸入特徵（或影像）→ 回傳 logits / 機率 / 類別
- `GET /health`：健康檢查

但內部有幾個教學上非常重要的點：

1) **模型只載入一次，不要每個 request 重新載入**
   - 把 `model = ...` 放在 module 最外層或 startup event

2) **推論要用 `eval()` + `torch.inference_mode()`**
   - 避免 dropout/bn 行為錯誤，也避免多餘的 gradient 記錄

3) **輸入/輸出格式要教清楚**（這是 DL 教學的重點）
   - 純數值：JSON array 最簡單（例如 `features: number[]`）
   - 圖片：教學上常用「base64（JSON）」或「multipart/form-data」

PyTorch 推論的骨架（概念示意）：

```python
import torch
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


class PredictRequest(BaseModel):
    features: list[float]


class PredictResponse(BaseModel):
    logits: list[float]


# 模型初始化：只做一次
model = ...  # torch.nn.Module
model.eval()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = torch.tensor(req.features, dtype=torch.float32).unsqueeze(0)  # (1, d)
    with torch.inference_mode():
        logits = model(x)
    return {"logits": logits.squeeze(0).tolist()}
```

如果你後續要做「訓練過程可視化」（loss 曲線、梯度下降動畫），通常會再加：

- `POST /train/step`：做一步訓練，回傳 loss、w/b、metrics
- 或 `GET /train/stream`：用 WebSocket / SSE 串流回傳訓練狀態

（這部分我可以再幫你寫成第三單元：把 gradient descent / training loop 變成可互動 UI。）

---

## 3) 前端 vs 後端：用 Python 友善的方式看架構

### 3.1 傳統 Python 後端架構（你熟悉的）

```
Browser Frontend
      |
      v
FastAPI / Flask (Python backend)
      |
      v
DL / ML Model (Python)
```

特點：

- Python 後端通常是核心：API、商業邏輯、資料處理、模型
- 前端可能是簡單頁面或獨立 SPA

### 3.2 現代 Next.js + Python DL 架構（視覺化專案常見）

```
Browser
  |
  v
Next.js Frontend (React UI)
  |
  v
Next.js API / BFF (Node runtime)
  |
  v
Python DL Service (FastAPI)
```

非常重要的澄清：

- 使用 Next.js **不代表放棄 Python**
- Python 仍然非常適合：
  - DL/ML 訓練與推論
  - NumPy/PyTorch 計算
  - 資料處理、視覺化前的數值運算

這個專案（線性模型）可以有兩種合理做法：

1) **全部在前端算（入門最直覺）**
   - 直接在 React 裡算：給定 x、w、b → 計算 y
   - 優點：速度快、最少依賴、容易理解

2) **前端視覺化 + Python 服務（為未來擴展鋪路）**
   - Next.js 呼叫 FastAPI
   - 優點：以後換成更複雜的模型也很自然（例如 training、推論）

---

## 4) macOS 環境安裝（Homebrew + Node.js）

### 4.1 什麼是 Homebrew？

**Homebrew 是 macOS 上很常用的套件管理工具（package manager）。**

你可以把它想成 macOS 版的「安裝器＋版本管理」：

- 用指令安裝：`brew install node`
- 用指令更新：`brew update`
- 用指令升級：`brew upgrade`

對 Python 使用者的類比：

- `pip`：管理 Python 套件（裝到 Python 環境裡）
- `brew`：管理系統層級工具（node、git、wget…）

### 4.2 安裝 Homebrew（如果尚未安裝）

在 Terminal 執行：

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

安裝後驗證：

```bash
brew -v
```

### 4.3 什麼是「install node」？你到底裝了什麼？

當你執行：

```bash
brew install node
```

通常會得到：

- `node`：Node.js runtime（讓 JS 在本機/伺服器執行）
- `npm`：Node 套件管理工具（通常會跟著 Node 一起安裝）

### 4.4 驗證 Node.js / npm

```bash
node -v
npm -v
```

如果能看到版本號（例如 `v20.x.x`），代表安裝成功。

---

## 5) npm 與 npx：它們是什麼？為什麼需要？

### 5.1 什麼是 npm？

**npm（Node Package Manager）是 Node 世界最常見的套件管理工具。**

你可以把它對照到：

- `pip install ...`（Python）
- `npm install ...`（Node/JS）

npm 的常見用途：

- 安裝依賴：React、Next.js、開發工具等
- 執行專案腳本：`npm run dev`、`npm run build`
- 管理依賴清單：`package.json`

### 5.2 什麼是 npx？

**npx 是用來「執行 Node 套件指令」的工具。**

它的核心價值是：

- 不需要你先全域安裝某個工具
- 可以「臨時下載、執行、用完就不一定要留著」

例子：

```bash
npx create-next-app@latest linear-visualizer
```

意思是：

- 使用最新版的 `create-next-app` 工具
- 幫你建立一個 Next.js 專案資料夾 `linear-visualizer`

對 Python 的類比（概念上，不是完全等價）：

- 你可能會用 `python -m some_tool ...` 或用一次性工具建立模板
- npx 的定位就是「一次性跑工具」

---

## 6) 建立 Next.js 專案（線性模型視覺化）

### 6.1 建立專案

在你想放專案的資料夾下：

```bash
npx create-next-app@latest linear-visualizer
```

接著進入專案：

```bash
cd linear-visualizer
```

### 6.2 建議選項（推薦）與概念解釋

建立時通常會問一些選項，這裡提供建議選擇與「為什麼」：

#### (1) TypeScript：建議選 Yes

你會得到：

- `.ts` / `.tsx` 檔案
- 更好的自動補全與型別檢查

對本專案的好處：

- `w`、`b`、`x`、`y` 都是數值，型別清楚可減少錯誤
- 未來加入資料結構（例如訓練資料點、loss 曲線）也更好維護

#### (2) App Router：建議選 Yes

概念上它是一種路由與專案結構：

- 用 `app/` 目錄來表示頁面與路由
- `app/page.tsx` 代表首頁 `/`

對初學者的好處：

- 結構更直覺：檔案在哪裡，路由就在哪裡
- 更容易把 UI 與資料取得方式整理在一起

#### (3) Tailwind CSS：建議選 Yes

概念上它是一套「用 class 組樣式」的方法。

對視覺化教學專案的好處：

- 排版清爽、容易做出一致的間距
- 不需要花太多時間在寫大量自訂 CSS

---

## 7) 生成的專案檔案結構解釋（你會看到哪些檔案？）

不同版本的 create-next-app 產生的細節可能略有差異，但常見會包含以下重點檔案/資料夾。

### 7.1 你一定會遇到的核心檔案

#### `package.json`

- 作用：定義專案名稱、腳本（scripts）、依賴（dependencies）
- 你會常用的指令通常在這裡（例如 `dev`, `build`, `start`）

你可以把它想成 Node 專案的「控制中心」。

#### `package-lock.json`（或 `pnpm-lock.yaml` / `yarn.lock`）

- 作用：鎖定依賴的精確版本，讓每個人安裝到一致的套件組合
- 通常不手動改它

#### `node_modules/`

- 作用：npm 安裝的套件會放在這
- 很大、通常不進 Git

對 Python 類比：

- 有點像你的 `.venv/` 或 site-packages（但方式不同）

### 7.2 Next.js 相關

#### `app/`

- 作用：App Router 的主要目錄
- 你會在這裡寫頁面與元件

常見內容：

- `app/page.tsx`：首頁 `/`
- `app/layout.tsx`：全站共用 layout（例如 header/footer）
- `app/globals.css`：全域 CSS（Tailwind 常會在這裡初始化）

#### `public/`

- 作用：靜態資源（圖片、favicon 等）
- 放在這裡的檔案可以用 `/xxx.png` 直接被存取

### 7.3 TypeScript / Tailwind / 工具設定

#### `tsconfig.json`

- 作用：TypeScript 編譯設定
- 一般初學可以先不碰，等需要調整路徑別名或嚴格程度再看

#### `tailwind.config.*`

- 作用：Tailwind 設定（例如掃描哪些檔案、擴充主題）

#### `postcss.config.*`

- 作用：CSS 處理管線設定（Tailwind 會用到）

#### `next.config.*`

- 作用：Next.js 的進階設定（例如圖片域名、重寫規則）

#### `eslint.config.*`（或 `.eslintrc.*`）

- 作用：程式碼風格與錯誤檢查
- 對新手好處：少踩坑，提前提醒不合理寫法

---

## 8) 執行專案（開發模式）

### 8.1 啟動開發伺服器

在專案資料夾內：

```bash
npm run dev
```

你可以把它理解成：

- 啟動 Next.js 的開發伺服器
- 監看檔案變更
- 你存檔後，瀏覽器會自動更新（hot reload）

### 8.2 用瀏覽器打開

通常預設是：

- http://localhost:3000

`localhost` 表示「你自己的電腦」，`3000` 是埠號（port）。

---

## 9) 這個專案在做什麼？（把 ML 概念對應到 UI）

### 9.1 線性模型的參數直覺

$$y = wx + b$$

- $w$：斜率（控制線的傾斜程度）
  - 例：$w$ 越大，線越「陡」
  - 例：$w$ 為負，線會往右下

- $b$：截距（控制線的上下平移）
  - 例：$b$ 變大，整條線往上移

### 9.2 建議的互動元素（教學上很有效）

- 兩個滑桿：調整 $w$ 與 $b$
- 一個公式區：即時顯示目前 $y = wx + b$ 的數值
- 一個圖表區：畫出直線（可以用 SVG/Canvas 或簡單圖表）
-（可選）顯示某幾個固定 x 點的 y 值，讓你看到數值怎麼變

這些互動會自然導向更深的 ML/DL：

- 加入資料點（x, y）
- 定義 loss（例如 MSE）
- 讓 w、b 透過 gradient descent 自動更新

---

## 10) Learning Outcomes（你完成後會得到什麼能力）

完成這個線性模型視覺化專案後，你會獲得：

- Web 基礎：HTML/CSS/JS 分工、瀏覽器與伺服器的概念
- React 互動核心：用 state 管理參數，狀態改變帶動畫面更新
- Next.js 入門全貌：App Router 的專案結構、開發伺服器、以及「同專案可包含 API/BFF」的觀念
- ML 直覺：參數 $w$、$b$ 如何影響模型輸出與函數圖形
- 後續 DL demo 的基礎：loss、gradient descent、可視化訓練過程

---

## 11) 下一步（如果你想接 Python / FastAPI）

當你準備把「視覺化前端」接上「Python 計算」時，常見做法是：

- Next.js：負責 UI + 互動 +（可選）BFF
- FastAPI：負責模型計算、資料處理、訓練或推論

你會得到一個分工清楚的全端架構：

- UI 反應快（前端即時計算或即時渲染）
- 重計算交給 Python（更適合 ML/DL 生態）

