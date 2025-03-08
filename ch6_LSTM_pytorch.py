"""
PyTorch LSTM 實作教學
==================

本章節介紹長短期記憶網路(LSTM)的實現，包括：
1. LSTM 架構原理
2. 時間序列預測實作
3. LSTM 門控機制
4. 模型訓練與評估
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from sklearn.preprocessing import MinMaxScaler

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class TimeSeriesLSTM(nn.Module):
    """LSTM時間序列預測模型"""
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2):
        super(TimeSeriesLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM層
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 全連接層
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        
        # 初始化隱藏狀態和細胞狀態
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM前向傳播
        out, _ = self.lstm(x, (h0, c0))
        # out shape: [batch_size, seq_len, hidden_dim]
        
        # 只使用最後一個時間步的輸出
        out = self.fc(out[:, -1, :])
        # out shape: [batch_size, output_dim]
        
        return out

class TimeSeriesDataset(Dataset):
    """時間序列數據集"""
    def __init__(self, data, seq_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.data) - self.seq_length
        
    def __getitem__(self, idx):
        # 獲取輸入序列和目標值
        sequence = self.data[idx:idx + self.seq_length].unsqueeze(-1)  # 添加特徵維度
        target = self.data[idx + self.seq_length]
        
        return sequence, target

def generate_time_series(n_samples=1000):
    """生成示例時間序列數據"""
    # 生成正弦波 + 噪聲
    t = np.linspace(0, 100, n_samples)
    series = np.sin(0.1 * t) + np.random.normal(0, 0.1, n_samples)
    
    # 數據標準化
    scaler = MinMaxScaler()
    series = scaler.fit_transform(series.reshape(-1, 1))
    
    return series, scaler

def plot_lstm_gates(save_path):
    """視覺化LSTM門控機制"""
    plt.figure(figsize=(15, 5))
    
    x = np.linspace(-5, 5, 100)
    sigmoid = 1 / (1 + np.exp(-x))
    tanh = np.tanh(x)
    
    plt.subplot(1, 3, 1)
    plt.plot(x, sigmoid, label='Sigmoid (Input/Forget/Output Gates)')
    plt.grid(True)
    plt.title('Gate Activation Functions')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(x, tanh, label='Tanh (Cell State)')
    plt.grid(True)
    plt.title('Cell State Activation')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(x, sigmoid * tanh, label='Gate * State')
    plt.grid(True)
    plt.title('Gated Information Flow')
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_memory_flow(save_path):
    """視覺化LSTM記憶流"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 繪製主要的記憶流
    ax.plot([0, 4], [0, 0], 'k-', linewidth=2, label='Cell State')
    
    # 添加門控
    gates = ['Input Gate', 'Forget Gate', 'Output Gate']
    positions = [1, 2, 3]
    colors = ['g', 'r', 'b']
    
    for gate, pos, color in zip(gates, positions, colors):
        # 繪製門控
        ax.plot([pos, pos], [-1, 1], f'{color}-', linewidth=2, label=gate)
        # 添加箭頭表示信息流
        ax.arrow(pos, 0, 0.2, 0, head_width=0.1, head_length=0.1, fc=color, ec=color)
    
    ax.set_xlim(0, 4)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    ax.set_title('LSTM Memory Flow')
    
    plt.savefig(save_path)
    plt.close()

class LSTMTrainer:
    """LSTM訓練器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'val_loss': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for sequences, targets in pbar:
            sequences = sequences.to(self.device)
            targets = targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(sequences)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss/(pbar.n+1)})
        
        return total_loss / len(train_loader)
    
    def evaluate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(sequences)
                loss = criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def plot_training_history(self):
        """繪製訓練歷史"""
        plt.figure(figsize=(10, 5))
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(SAVE_DIR, 'ch6_lstm_training_history.png'))
        plt.close()

def plot_time_series_prediction(true_values, predictions, save_path):
    """繪製時間序列預測結果"""
    plt.figure(figsize=(12, 6))
    plt.plot(true_values, label='True Values', color='blue')
    plt.plot(predictions, label='Predictions', color='red', linestyle='--')
    plt.title('Time Series Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 視覺化LSTM組件
    plot_lstm_gates(os.path.join(SAVE_DIR, 'ch6_lstm_gates.png'))
    plot_memory_flow(save_path=os.path.join(SAVE_DIR, 'ch6_lstm_memory_flow.png'))
    
    # 生成時間序列數據
    series, scaler = generate_time_series()
    
    # 設置序列長度和數據集參數
    seq_length = 20
    train_size = int(len(series) * 0.8)
    
    # 創建數據集
    train_dataset = TimeSeriesDataset(series[:train_size], seq_length)
    val_dataset = TimeSeriesDataset(series[train_size:], seq_length)
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(val_dataset, batch_size=1)
    
    # 創建模型
    model = TimeSeriesLSTM(
        input_dim=1,
        hidden_dim=64,
        num_layers=2,
        output_dim=1
    )
    
    # 創建訓練器和優化器
    trainer = LSTMTrainer(model, device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    num_epochs = 50
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss = trainer.evaluate(val_loader, criterion)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['val_loss'].append(val_loss)
        
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
    
    # 繪製訓練歷史
    trainer.plot_training_history()
    
    # 生成預測結果
    model.eval()
    with torch.no_grad():
        predictions = []
        for batch in test_loader:
            sequences, _ = batch
            sequences = sequences.to(device)
            pred = model(sequences)
            predictions.extend(pred.cpu().numpy())
    
    # 繪製預測結果
    true_values = series[train_size:][:len(predictions)]
    plot_time_series_prediction(
        true_values, predictions,
        os.path.join(SAVE_DIR, 'ch6_lstm_predictions.png')
    )
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ch6_time_series_lstm.pth'))
    print("\n模型和圖片已保存到", SAVE_DIR)

if __name__ == '__main__':
    main()
