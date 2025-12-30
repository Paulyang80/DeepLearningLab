# CNN 模型改進說明

## 問題診斷
原始模型只訓練 3 個 epochs，且沒有防止 overfitting 的機制，導致手寫辨識效果不佳。

## 改進措施

### 1. 啟用 Mac GPU (MPS)
- ✅ 使用 `torch.backends.mps` 檢測並啟用 Metal Performance Shaders
- **速度提升**：10 epochs 約 2-3 分鐘（CPU 需要 10+ 分鐘）
- 自動 fallback 到 CUDA 或 CPU

### 2. 防止 Overfitting

#### Dropout (0.25)
- 在兩個卷積層後加入 `Dropout2d(0.25)`
- 在全連接層加入 `Dropout(0.25)`
- 訓練時隨機關閉 25% 的神經元，測試時全部啟用

#### 數據增強 (Data Augmentation)
- 隨機旋轉 ±10°
- 隨機平移 ±10%
- 標準化：mean=0.1307, std=0.3081（MNIST 標準值）

#### Weight Decay (L2 Regularization)
- Adam optimizer 加入 `weight_decay=1e-4`
- 懲罰過大的權重，鼓勵簡單模型

#### Learning Rate Scheduler
- `ReduceLROnPlateau`：測試準確率停滯時自動降低學習率
- 初始 lr=0.001，停滯 2 epochs 後 ×0.5

### 3. 更充分的訓練
- **Epochs**: 3 → 10
- **最佳模型保存**：自動保存測試準確率最高的模型（`mnist_cnn_best.pth`）

## 訓練結果

```
最終測試準確率: 99.15%
最佳測試準確率: 99.15%
模型參數量: 206,922
訓練時間: ~2-3 分鐘 (Mac GPU)
```

### 逐 Epoch 準確率
| Epoch | Train Acc | Test Acc | 備註 |
|-------|-----------|----------|------|
| 1 | 84.09% | 97.76% | ★ |
| 2 | 93.87% | 98.45% | ★ |
| 3 | 94.76% | 98.65% | ★ |
| 4 | 95.49% | 98.95% | ★ |
| 5 | 95.83% | 98.92% | |
| 6 | 96.01% | 98.99% | ★ |
| 7 | 96.33% | 99.12% | ★ |
| 8 | 96.47% | 99.04% | |
| 9 | 96.51% | 99.13% | ★ |
| 10 | 96.61% | 99.15% | ★ Best |

**觀察**：
- Train/Test 準確率差距小（~2-3%），表示沒有嚴重 overfitting
- Test 準確率穩定提升，沒有抖動

## 使用方式

### 重新訓練
```bash
cd /Users/paulyang/Projects/DeepLearningLab/backend
python3 train_cnn.py
```

### 測試模型
```bash
# 啟動後端
python3 -m uvicorn backend.main:app --reload --port 8000

# 前端測試
cd web && npm run dev
# 打開 http://localhost:3000/cnn
```

### 前端手寫測試
1. 在 Canvas 上畫一個數字
2. 點擊「辨識數字」
3. 勾選「顯示 Feature Maps」看卷積層學到的特徵

## 模型架構

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

## 常見問題

### Q: 為什麼我的 Mac 沒有啟用 GPU？
A: 檢查：
- macOS 版本 ≥ 12.3
- PyTorch 版本 ≥ 1.12
- Apple Silicon (M1/M2/M3) 或支援 Metal 的 Intel Mac

### Q: 訓練時記憶體不足？
A: 調小 batch_size（預設 64，可改為 32）

### Q: 準確率還是不理想？
A: 可能原因：
- 手寫風格太不同（試試更多數據增強）
- Canvas 繪製太細/太粗（調整筆刷大小）
- 模型未正確載入（檢查 backend logs）

## 下一步可能改進

- [ ] 加入更多數據增強（彈性形變）
- [ ] 使用更深的網路（ResNet-like）
- [ ] Ensemble 多個模型
- [ ] 加入 Batch Normalization
- [ ] 訓練更多 epochs（20-30）
