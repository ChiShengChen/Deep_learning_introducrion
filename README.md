# PyTorch 深度學習入門教程

這個教程專為深度學習初學者設計，循序漸進地介紹 PyTorch 的基礎知識到進階應用。

## 環境設置

基本要求：
- Python 3.8+
- PyTorch 2.0+
- CUDA (可選，用於 GPU 加速)

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
  - 激勵函數選擇

- Chapter 4: 卷積神經網路 (CNN)
  - CNN 基本原理
  - 圖像分類實作
  - MNIST 手寫數字識別

### 序列模型
- Chapter 5: 循環神經網路 (RNN)
  - RNN 基本原理
  - 隨機生成文本分類實作

- Chapter 6: 長短期記憶網路 (LSTM)
  - LSTM 架構解析
  - 隨機生成序列預測實作

- Chapter 7: 門控循環單元 (GRU)
  - GRU vs LSTM
  - 隨機生成序列預測實作

- Chapter 8: Transformer
  - 注意力機制
  - 自注意力機制
  - AG_NEWS 文本分類實作

- Chapter 9: Mamba
  - 狀態空間機制
  - AG_NEWS 文本分類實作

## 環境配置
- Python 3.7+
- PyTorch 2.0+
- torchvision
- torchtext

## Dataset說明
### MNIST手寫數字
MNIST 是一個經典的手寫數字識別數據集，讓我簡單介紹一下：  
內容：  
- 手寫數字（0-9）的灰度圖像
- 每張圖像大小為 28x28 像素
- 每個像素值範圍 0-255（灰度值）  
規模：  
- 訓練集：60,000 張圖像
- 測試集：10,000 張圖像
- 總共 70,000 張圖像
類別：  
- 10 個類別（數字 0-9）
- 每個類別數量大致平衡

### AG News 
AG News 是一個新聞文本分類數據集，包含四個類別：  
- World (世界新聞)
- Sports (體育新聞)
- Business (商業新聞)
- Sci/Tech (科技新聞)  
數據集規模：    
- 訓練集：120,000 條新聞文本
- 測試集：7,600 條新聞文本  
每條數據包含：  
- 新聞標題
- 新聞描述
- 類別標籤 (1-4)

## 學習建議
1. 按照章節順序學習
2. 確保理解每個概念後再進入下一章
3. 動手實踐每個範例
4. 嘗試修改參數觀察結果

## 參考資源
- [PyTorch 官方文檔](https://pytorch.org/docs/stable/index.html)
- [PyTorch 教程](https://pytorch.org/tutorials/)
