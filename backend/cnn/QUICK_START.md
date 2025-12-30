# CNN 模組快速使用指南

## 📍 所有腳本現在都在 `backend/cnn/` 目錄中

### 🎯 訓練模型

```bash
cd backend/cnn
python3 train_cnn.py
```

**輸出**：
- `../models/mnist_cnn.pth` - 最終模型
- `../models/mnist_cnn_best.pth` - 最佳模型
- 測試準確率：99%+
- 訓練時間：~2-3 分鐘 (Mac GPU)

### 🧪 測試模組

```bash
cd backend/cnn
python3 test_cnn_module.py
```

測試項目：
- ✓ 模型導入
- ✓ 預測器初始化
- ✓ 模型載入
- ✓ 模型資訊
- ✓ 圖片預處理

### 🔍 測試 API

確保後端運行中，然後：

```bash
cd backend/cnn
python3 test_mnist_api.py
```

這會：
- 從 MNIST 測試集讀取真實圖片
- 發送到 API 進行預測
- 比對預測結果與真實標籤
- 顯示準確率

### 🐛 診斷問題

#### 診斷預處理
```bash
cd backend/cnn
python3 diagnose_preprocess.py
```

檢查：
- MNIST 原始圖片格式
- 訓練時的轉換
- API 預處理結果
- 是否需要顏色反轉

#### 診斷網頁問題
```bash
cd backend/cnn
python3 diagnose_web_issue.py
```

測試：
- 後端是否運行
- 模型是否載入
- API 預測是否正常
- 直接模組調用是否正常

## 📂 目錄結構

```
backend/
├── cnn/                          ← 所有 CNN 相關檔案都在這裡
│   ├── __init__.py
│   ├── model.py                 ← SimpleCNN 定義
│   ├── predictor.py             ← CNNPredictor 類別
│   ├── trainer.py               ← CNNTrainer 類別
│   ├── README.md                ← 完整文檔
│   │
│   ├── train_cnn.py             ← 訓練腳本
│   ├── test_mnist_api.py        ← API 測試
│   ├── test_cnn_module.py       ← 模組測試
│   ├── diagnose_preprocess.py   ← 預處理診斷
│   ├── diagnose_web_issue.py    ← 網頁問題診斷
│   │
│   └── docs/                    ← 詳細文檔
│       ├── README.md
│       ├── IMPROVEMENTS.md
│       └── BUGFIX_PREDICTION.md
│
├── models/                       ← 訓練好的模型
│   ├── mnist_cnn.pth
│   ├── mnist_cnn_best.pth
│   └── .gitignore
│
└── main.py                       ← FastAPI 後端（使用 cnn 模組）
```

## 🚀 啟動後端

```bash
# 從專案根目錄
cd /Users/paulyang/Projects/DeepLearningLab
python3 -m uvicorn backend.main:app --reload --port 8000
```

## 🌐 啟動前端

```bash
# 從 web 目錄
cd web
npm run dev
```

然後訪問：http://localhost:3000/cnn

## ⚠️ 常見問題

### 問題：執行腳本時 "No module named 'backend'"

**原因**：Python 找不到 `backend` 模組

**解決方案**：
所有腳本已經設置了正確的路徑，確保從 `backend/cnn/` 目錄執行：

```bash
cd /Users/paulyang/Projects/DeepLearningLab/backend/cnn
python3 train_cnn.py  # ✓ 正確
```

**不要**從其他目錄執行：
```bash
cd /Users/paulyang/Projects/DeepLearningLab/backend
python3 train_cnn.py  # ✗ 錯誤（腳本已移到 cnn/）
```

### 問題：網頁預測全部返回 0

**排查步驟**：

1. **重新訓練模型**
   ```bash
   cd backend/cnn
   python3 train_cnn.py
   ```

2. **測試 API**
   ```bash
   cd backend/cnn
   python3 test_mnist_api.py
   ```
   應該顯示 95%+ 準確率

3. **前端硬刷新**
   - Chrome/Edge: Cmd+Shift+R (Mac)
   - Safari: Cmd+Option+R

4. **檢查後端日誌**
   應該看到：`✓ Loaded CNN model from ...`

### 問題：找不到 MNIST 資料

確認以下目錄存在：
```
notebooks/data/MNIST/raw/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

## 📚 更多資訊

- 完整文檔：`backend/cnn/README.md`
- 改進記錄：`backend/cnn/docs/IMPROVEMENTS.md`
- Bug 修復：`backend/cnn/docs/BUGFIX_PREDICTION.md`

---

**最後更新**: 2025-12-30
