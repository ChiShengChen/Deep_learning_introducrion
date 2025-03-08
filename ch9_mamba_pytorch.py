"""
PyTorch Mamba 實作教學
===================

本章節介紹 Mamba 模型的實現，包括：
1. 狀態空間模型基礎
2. 選擇性狀態空間 (S4)
3. 硬體加速優化
4. 文本分類實作
5. 與 Transformer 比較
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
import math

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class CausalConv1d(nn.Module):
    """因果卷積層"""
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels,
            kernel_size,
            padding=self.padding
        )
    
    def forward(self, x):
        # 應用卷積
        x = self.conv(x)
        # 移除多餘的填充（只保留因果部分）
        if self.padding != 0:
            x = x[:, :, :-self.padding]
        return x

class SelectiveSSM(nn.Module):
    """選擇性狀態空間模塊"""
    def __init__(self, d_model, d_state, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        # S4 參數
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_model, d_state))
        self.C = nn.Parameter(torch.randn(d_state, d_model))
        
        # 選擇性注意力
        self.dt = nn.Parameter(torch.randn(d_model))
        self.dropout = nn.Dropout(dropout)
        
        # 因果卷積
        self.conv = CausalConv1d(d_model, d_model, kernel_size=3)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        B, L, D = x.shape
        
        # 狀態空間更新
        h = torch.zeros(B, self.d_state, device=x.device)
        output = []
        
        for t in range(L):
            # 更新狀態
            h = torch.tanh(
                torch.matmul(h, self.A) + 
                torch.matmul(x[:, t, :], self.B)
            )
            # 生成輸出
            y = torch.matmul(h, self.C)
            output.append(y)
        
        output = torch.stack(output, dim=1)
        
        # 應用選擇性注意力
        dt = torch.sigmoid(self.dt)
        output = output * dt.view(1, 1, -1)
        
        # 應用因果卷積
        output = self.conv(output.transpose(1, 2)).transpose(1, 2)
        
        return self.dropout(output)

class MambaClassifier(nn.Module):
    """Mamba文本分類模型"""
    def __init__(self, vocab_size, d_model, d_state, num_layers, num_classes, dropout=0.1):
        super().__init__()
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Mamba層
        self.layers = nn.ModuleList([
            SelectiveSSM(d_model, d_state, dropout)
            for _ in range(num_layers)
        ])
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_classes)
        )
    
    def forward(self, x):
        # 詞嵌入
        x = self.embedding(x)
        
        # Mamba層
        for layer in self.layers:
            x = layer(x)
        
        # 池化
        x = x.mean(dim=1)
        
        # 分類
        return self.classifier(x)

def plot_state_space(save_path):
    """視覺化狀態空間模型"""
    plt.figure(figsize=(12, 6))
    
    # 生成示例數據
    t = np.linspace(0, 10, 100)
    state = np.sin(t) * np.exp(-0.1 * t)
    output = np.tanh(state)
    
    plt.subplot(1, 2, 1)
    plt.plot(t, state, label='State')
    plt.plot(t, output, label='Output')
    plt.title('State Space Evolution')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(state, output)
    plt.title('State-Output Relationship')
    plt.xlabel('State')
    plt.ylabel('Output')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_selective_scan(save_path):
    """視覺化選擇性掃描"""
    plt.figure(figsize=(10, 6))
    
    # 生成示例數據
    x = np.linspace(0, 1, 100)
    y1 = np.sin(2 * np.pi * x)
    y2 = np.cos(2 * np.pi * x)
    # 使用 1/(1 + exp(-x)) 來實現 sigmoid
    selection = 1 / (1 + np.exp(-5 * (x - 0.5)))
    
    plt.plot(x, y1, label='Input 1', alpha=0.5)
    plt.plot(x, y2, label='Input 2', alpha=0.5)
    plt.plot(x, selection, label='Selection', color='red')
    plt.plot(x, y1 * selection + y2 * (1-selection), label='Output', color='purple')
    
    plt.title('Selective State Space Scanning')
    plt.xlabel('Position')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

class MambaTrainer:
    """Mamba訓練器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for texts, labels in pbar:
            texts = texts.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(texts)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': total_loss/(pbar.n+1),
                'acc': 100.*correct/total
            })
        
        return total_loss/len(train_loader), 100.*correct/total
    
    def evaluate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(texts)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return total_loss/len(val_loader), 100.*correct/total
    
    def plot_training_history(self):
        """繪製訓練歷史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'ch9_mamba_training_history.png'))
        plt.close()

def main():
    # 設置參數
    d_model = 256
    d_state = 64
    num_layers = 4
    num_classes = 4  # AG_NEWS有4個類別
    max_len = 256   # 文本長度
    batch_size = 32
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 視覺化組件
    plot_state_space(os.path.join(SAVE_DIR, 'ch9_state_space.png'))
    plot_selective_scan(os.path.join(SAVE_DIR, 'ch9_selective_scan.png'))
    
    # 加載AG_NEWS數據集
    from ch8_transformer_pytorch import load_ag_news, TextDataset
    (train_texts, train_labels), (test_texts, test_labels), vocab, tokenizer = load_ag_news()
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Training samples: {len(train_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # 創建數據集
    train_dataset = TextDataset(train_texts, train_labels, vocab, tokenizer, max_len)
    test_dataset = TextDataset(test_texts, test_labels, vocab, tokenizer, max_len)
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # 創建模型
    model = MambaClassifier(
        vocab_size=len(vocab),
        d_model=d_model,
        d_state=d_state,
        num_layers=num_layers,
        num_classes=num_classes
    )
    
    # 創建訓練器和優化器
    trainer = MambaTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        test_loss, test_acc = trainer.evaluate(test_loader, criterion)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_loss'].append(test_loss)
        trainer.history['val_acc'].append(test_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # 繪製訓練歷史
    trainer.plot_training_history()
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ch9_mamba_classifier.pth'))
    print("\n模型和圖片已保存到", SAVE_DIR)

if __name__ == '__main__':
    main() 