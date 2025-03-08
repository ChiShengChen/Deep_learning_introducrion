# PyTorch 深度學習入門教程

這個教程專為深度學習初學者設計，循序漸進地介紹 PyTorch 的基礎知識到進階應用。

## 環境設置

基本要求：
- Python 3.8+
- PyTorch 2.0+
- CUDA (可選，用於 GPU 加速)

## 章節內容

### Chapter 5: RNN (循環神經網路)
使用 RNN 實現文本分類。

需要安裝的套件：
```
pip install torch torchtext numpy matplotlib tqdm scikit-learn pandas
```

## 課程大綱

### 基礎知識
- Chapter 0: PyTorch 基礎介紹
  - 安裝指南
  - 張量(Tensor)基礎操作
  - 自動微分(Autograd)

- Chapter 1: 資料處理基礎
  - Dataset 和 DataLoader
  - 資料預處理和轉換
  - 批次處理

- Chapter 2: 模型訓練基礎
  - 訓練循環
  - 驗證循環
  - 模型評估

### 神經網絡模型
- Chapter 3: 自定義神經網路
  - 神經網路基本組件
  - 模型架構設計
  - 啟動函數選擇

- Chapter 4: 卷積神經網路 (CNN)
  - CNN 基本原理
  - 圖像分類實作
  - MNIST 手寫數字識別

### 序列模型
- Chapter 5: 循環神經網路 (RNN)
  - RNN 基本原理
  - 文本分類實作

- Chapter 6: 長短期記憶網路 (LSTM)
  - LSTM 架構解析
  - 序列預測實作

- Chapter 7: 門控循環單元 (GRU)
  - GRU vs LSTM
  - 序列預測實作

- Chapter 8: Transformer
  - 注意力機制
  - 自注意力機制
  - 文本分類實作

## 環境配置
- Python 3.7+
- PyTorch 2.0+
- torchvision
- torchtext

## 學習建議
1. 按照章節順序學習
2. 確保理解每個概念後再進入下一章
3. 動手實踐每個範例
4. 嘗試修改參數觀察結果

## 參考資源
- [PyTorch 官方文檔](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)