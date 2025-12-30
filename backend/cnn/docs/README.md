# CNN/MNIST Backend

這個資料夾包含 MNIST 手寫數字辨識的 CNN 模型與訓練腳本。

## 模型架構

`SimpleCNN`：輕量級卷積神經網路
- Conv1: 1→16 channels (3×3 kernel, padding=1)
- ReLU + MaxPool(2×2)
- Conv2: 16→32 channels (3×3 kernel, padding=1)
- ReLU + MaxPool(2×2)
- FC1: 32×7×7 → 128
- FC2: 128 → 10 (classes 0-9)

## 訓練模型

第一次使用前需要先訓練模型：

```bash
cd /Users/paulyang/Projects/DeepLearningLab/backend
python3 train_cnn.py
```

這會：
- 使用 `../notebooks/data/MNIST` 中的資料（已存在）
- 訓練 3 個 epochs（約 1-2 分鐘，視硬體而定）
- 儲存模型到 `backend/models/mnist_cnn.pth`
- 預期測試準確率：~98%

## API Endpoints

### `POST /predict/mnist`

輸入：
```json
{
  "image": "data:image/png;base64,...",
  "return_feature_maps": false
}
```

輸出：
```json
{
  "predicted_digit": 7,
  "probabilities": [0.001, 0.002, ..., 0.95],
  "logits": [-5.2, -4.1, ..., 3.8],
  "conv1_maps": null  // 若 return_feature_maps=true 會回傳
}
```

### `GET /model/cnn/info`

回傳模型基本資訊（架構、路徑、是否已載入）。

## 測試

確認後端運行：
```bash
curl -sS http://localhost:8000/model/cnn/info
```

## 故障排除

- **模型未載入**：先執行 `python3 train_cnn.py`
- **找不到 MNIST 資料**：確認 `notebooks/data/MNIST/raw/` 中有資料檔案
- **記憶體不足**：調小 train_cnn.py 中的 batch_size

## 前端使用

在 `/cnn` 頁面：
1. 在 Canvas 上手繪數字
2. 點擊「辨識數字」
3. 查看預測結果與機率分佈
4. （可選）勾選「顯示 Feature Maps」看第一層卷積學到的特徵
