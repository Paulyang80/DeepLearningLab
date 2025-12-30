# CNN Module for MNIST Digit Recognition

這個模組提供完整的 MNIST 手寫數字辨識功能，包括模型定義、訓練和預測。

## 📁 目錄結構

```
backend/cnn/
├── __init__.py          # 模組初始化
├── model.py             # SimpleCNN 架構定義
├── predictor.py         # CNNPredictor 預測器
├── trainer.py           # CNNTrainer 訓練器
└── docs/                # 相關文檔
    ├── README.md        # 使用說明
    ├── IMPROVEMENTS.md  # 模型改進記錄
    └── BUGFIX_PREDICTION.md  # Bug 修復記錄
```

## 🏗️ 模型架構

**SimpleCNN**: 輕量級卷積神經網路

```
Input (1, 28, 28)
    ↓
Conv1: 1→16 (3×3, pad=1) + ReLU + MaxPool(2×2) + Dropout(0.25)
    ↓ (16, 14, 14)
Conv2: 16→32 (3×3, pad=1) + ReLU + MaxPool(2×2) + Dropout(0.25)
    ↓ (32, 7, 7)
Flatten → FC1: 1568→128 + ReLU + Dropout(0.25)
    ↓
FC2: 128→10 (logits)
    ↓
Softmax → 機率分佈
```

- **參數量**: ~207k
- **測試準確率**: 99%+
- **訓練時間**: ~2-3 分鐘 (Mac GPU)

## 🚀 快速開始

### 1. 訓練模型

第一次使用前需要先訓練模型：

```bash
cd /Users/paulyang/Projects/DeepLearningLab/backend
python3 train_cnn.py
```

這會：
- 使用 `../notebooks/data/MNIST` 中的資料
- 自動偵測並使用 Mac GPU (MPS) 或 CUDA
- 訓練 10 個 epochs
- 儲存模型到 `backend/models/mnist_cnn_best.pth` 和 `mnist_cnn.pth`
- 預期測試準確率：99%+

### 2. 使用預測器

```python
from backend.cnn import CNNPredictor
from pathlib import Path

# 初始化預測器
model_dir = Path("backend/models")
predictor = CNNPredictor(model_dir)

# 預測（base64 或 data URL）
result = predictor.predict(
    image_str="data:image/png;base64,...",
    return_feature_maps=True  # 可選：返回 feature maps
)

print(f"預測數字: {result['predicted_digit']}")
print(f"機率: {result['probabilities']}")
```

### 3. API 使用

啟動後端後，可使用以下 endpoints：

#### `POST /predict/mnist`

**Request:**
```json
{
  "image": "data:image/png;base64,...",
  "return_feature_maps": false
}
```

**Response:**
```json
{
  "predicted_digit": 7,
  "probabilities": [0.001, 0.002, ..., 0.95],
  "logits": [-5.2, -4.1, ..., 3.8],
  "conv1_maps": null
}
```

#### `GET /model/cnn/info`

返回模型基本資訊（架構、路徑、是否已載入等）。

## 🎯 主要特性

### 防止 Overfitting
- **Dropout** (0.25)：訓練時隨機關閉 25% 神經元
- **數據增強**：隨機旋轉 ±10°、平移 ±10%
- **Weight Decay** (L2)：L2 正則化 (1e-4)
- **Learning Rate Scheduler**：測試準確率停滯時自動降低學習率

### Mac GPU 加速
- 自動偵測並啟用 Metal Performance Shaders (MPS)
- 訓練速度提升 5-10 倍
- 10 epochs 僅需 2-3 分鐘

### Feature Map 視覺化
- 可提取 Conv1 (16個) 和 Conv2 (32個) feature maps
- 用於理解模型學到的特徵

## 📊 訓練結果

```
最終測試準確率: 99.15%
最佳測試準確率: 99.15%
模型參數量: 206,922
訓練時間: ~2-3 分鐘 (Mac GPU)
```

Train/Test 準確率差距小（~2-3%），表示沒有嚴重 overfitting。

## 🔧 自訂訓練

使用 `CNNTrainer` 類別進行客製化訓練：

```python
from backend.cnn.trainer import CNNTrainer
from pathlib import Path

trainer = CNNTrainer(
    data_root=Path("notebooks/data"),
    model_dir=Path("backend/models"),
    batch_size=64,
    lr=0.001,
    dropout_rate=0.25,
    weight_decay=1e-4
)

model, best_acc = trainer.train(epochs=10, verbose=True)
print(f"Best accuracy: {best_acc:.2f}%")
```

## 📝 預處理細節

### MNIST 標準化
- **Mean**: 0.1307
- **Std**: 0.3081
- 這些值是從 MNIST 訓練集計算出來的

### 重要提醒
- MNIST 原始資料：**白底 (0) + 黑字 (高值)**
- Canvas 繪製：**白底 (#ffffff) + 黑筆 (#000000)**
- **不需要顏色反轉**！

訓練與推論必須使用完全相同的預處理流程。

## 🐛 故障排除

### 模型未載入
```bash
# 先執行訓練
python3 backend/train_cnn.py
```

### 找不到 MNIST 資料
確認 `notebooks/data/MNIST/raw/` 中有以下檔案：
- `train-images-idx3-ubyte`
- `train-labels-idx1-ubyte`
- `t10k-images-idx3-ubyte`
- `t10k-labels-idx1-ubyte`

### 記憶體不足
調小 batch_size：
```python
trainer = CNNTrainer(batch_size=32)  # 預設 64
```

### 預測總是返回 0
檢查：
1. 模型是否正確載入（查看 backend logs）
2. Canvas 設定是否正確（白底 + 黑筆）
3. 預處理是否與訓練一致

詳見 `docs/BUGFIX_PREDICTION.md`。

## 📚 相關文檔

- **IMPROVEMENTS.md**: 模型改進過程，包括 Mac GPU 使用、防 overfitting 技術
- **BUGFIX_PREDICTION.md**: 預測問題的診斷與修復過程
- **README.md** (models/): 原始的模型說明文件

## 🎨 前端整合

前端位於 `web/app/cnn/page.tsx`：
1. 在 Canvas 上手繪數字
2. 點擊「辨識數字」
3. 查看預測結果與機率分佈
4. （可選）勾選「顯示 Feature Maps」看卷積層特徵

訪問 http://localhost:3000/cnn 進行測試。

## 📄 License

MIT

---

**作者**: Paul Yang  
**更新時間**: 2025-12-30
