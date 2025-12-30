# 預測問題修復記錄

## 問題描述
模型總是預測數字 0，無論輸入什麼圖片。

## 根本原因
**訓練與推論的預處理不一致**

### 訓練時的預處理
```python
transforms.Compose([
    transforms.ToTensor(),  # 自動轉換到 [0,1]
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 標準化
])
```

### 原始推論預處理（錯誤）
```python
arr = np.array(img, dtype=np.float32) / 255.0  # 只做了 [0,1] 轉換
tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
# ❌ 缺少 Normalize 步驟！
```

## 解決方案

### 修正後的預處理
```python
# 1. 轉換到 [0, 1]
arr = arr / 255.0

# 2. 套用與訓練相同的標準化
mean = 0.1307
std = 0.3081
arr = (arr - mean) / std

# 3. 轉為 tensor
tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
```

### 關於顏色反轉
一開始懷疑需要顏色反轉（因為 Canvas 是黑底白字），但診斷後發現：
- MNIST 原始資料：白底（0）+ 黑字（高值）
- Canvas 已經設定：白底（#ffffff）+ 黑筆（#000000）
- **不需要反轉**

## 驗證結果

### 測試 1：單張圖片
```
True label: 7
Predicted: 7 ✓
Confidence: 99.95%
```

### 測試 2：20 張圖片
```
Accuracy: 19/20 = 95.0%
```

只有 1 張錯誤（3 被認成 8），這在可接受範圍內。

## 診斷工具

創建了兩個診斷腳本：

1. **`diagnose_preprocess.py`**
   - 比對不同預處理方法
   - 確認是否需要顏色反轉
   - 計算與訓練轉換的差異

2. **`test_mnist_api.py`**
   - 用真實 MNIST 圖片測試 API
   - 批量測試準確率
   - 顯示 top-3 預測機率

## 關鍵學習

1. **訓練與推論必須使用完全相同的預處理**
   - 包括 normalization 的 mean 和 std
   - 包括數據格式（顏色、尺寸）

2. **診斷比猜測更有效**
   - 用腳本驗證每個假設
   - 比對實際數值而非憑直覺

3. **MNIST 的標準值要記住**
   - mean = 0.1307
   - std = 0.3081
   - 這些值是從訓練集計算出來的統計量

## 前端使用提醒

現在 Canvas 設定正確：
- 白色背景（`#ffffff`）
- 黑色筆刷（`#000000`）
- 筆刷大小 12px（模擬手寫粗細）

使用者在 `/cnn` 頁面手寫數字時，應該能獲得準確的預測結果。
