"""
PyTorch Transformer 實作教學
========================

本章節介紹 Transformer 模型的實現，包括：
1. Transformer 架構原理
2. 自注意力機制實作
3. 多頭注意力機制
4. 位置編碼
5. 文本分類實戰
"""

import torchtext
import os
torchtext.disable_torchtext_deprecation_warning()

# 添加必要的導入
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class PositionalEncoding(nn.Module):
    """位置編碼層"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerClassifier(nn.Module):
    """Transformer文本分類模型"""
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, num_classes, max_len, dropout=0.1):
        super(TransformerClassifier, self).__init__()
        
        # 詞嵌入層
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        
        # Transformer編碼器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 分類頭
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_classes)
        )
        
        self.d_model = d_model
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                layer.bias.data.zero_()
                layer.weight.data.uniform_(-initrange, initrange)
    
    def forward(self, src, src_mask=None):
        # src shape: [batch_size, seq_len]
        
        # 詞嵌入和位置編碼
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        
        # Transformer編碼
        output = self.transformer_encoder(src, src_mask)
        
        # 使用序列的平均值進行分類
        output = output.mean(dim=1)
        
        # 分類
        output = self.classifier(output)
        return output

def generate_text_data(num_samples=1000):
    """生成示例文本數據"""
    # 創建一個簡單的詞彙表
    vocab_words = ['good', 'bad', 'happy', 'sad', 'movie', 'film', 'great', 'terrible', 
                  'excellent', 'poor', 'amazing', 'awful', 'best', 'worst', 'love', 'hate']
    
    # 創建詞彙表字典
    vocab = {word: i+1 for i, word in enumerate(vocab_words)}
    vocab['<unk>'] = 0
    vocab['<pad>'] = len(vocab)
    
    # 創建分詞器
    def simple_tokenizer(text):
        return text.lower().split()
    
    # 生成隨機文本
    texts = []
    labels = []
    for _ in range(num_samples):
        # 隨機選擇5-15個詞
        length = np.random.randint(5, 15)
        text = ' '.join(np.random.choice(vocab_words, size=length))
        label = np.random.randint(0, 2)  # 二分類標籤
        texts.append(text)
        labels.append(label)
    
    return (texts[:800], labels[:800]), (texts[800:], labels[800:]), vocab, simple_tokenizer

class TextDataset(Dataset):
    """文本數據集"""
    def __init__(self, texts, labels, vocab, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # 分詞並轉換為索引
        tokens = self.tokenizer(text)
        indices = [self.vocab[token] for token in tokens]
        
        # 填充或截斷
        if len(indices) < self.max_len:
            indices += [self.vocab['<pad>']] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices), torch.tensor(label)

def plot_positional_encoding(max_len=100, d_model=512, save_path=None):
    """視覺化位置編碼"""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    plt.figure(figsize=(15, 8))
    plt.pcolormesh(pe.numpy(), cmap='RdBu')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Sequence Position')
    plt.colorbar(label='Encoding Value')
    plt.title('Positional Encoding Matrix')
    
    plt.savefig(save_path)
    plt.close()

def plot_attention_weights(attention_weights, save_path):
    """視覺化注意力權重"""
    plt.figure(figsize=(10, 8))
    plt.imshow(attention_weights, cmap='viridis')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Position')
    plt.ylabel('Query Position')
    plt.title('Self-Attention Weight Matrix')
    
    plt.savefig(save_path)
    plt.close()

def plot_multi_head_attention(attention_weights, num_heads, save_path):
    """視覺化多頭注意力"""
    fig, axes = plt.subplots(2, num_heads//2, figsize=(15, 8))
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(attention_weights[i], cmap='viridis')
        ax.set_title(f'Attention Head {i+1}')
        fig.colorbar(im, ax=ax)
    
    plt.suptitle('Multi-Head Attention Visualization')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

class TransformerTrainer:
    """Transformer模型訓練器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
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
                'loss': total_loss/total,
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
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'ch8_transformer_training_history.png'))
        plt.close()

def load_ag_news():
    """加載AG_NEWS數據集"""
    import urllib.request
    import csv
    
    # 下載數據
    train_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
    test_url = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"
    
    def download_dataset(url):
        response = urllib.request.urlopen(url)
        lines = [line.decode('utf-8') for line in response.readlines()]
        texts = []
        labels = []
        for line in lines:
            label, title, description = line.strip().split('","')
            label = int(label.strip('"'))
            text = title + " " + description.strip('"')
            texts.append(text)
            labels.append(label - 1)  # 將標籤從1-4轉換為0-3
        return texts, labels
    
    print("Downloading training data...")
    train_texts, train_labels = download_dataset(train_url)
    print("Downloading test data...")
    test_texts, test_labels = download_dataset(test_url)
    
    # 初始化分詞器
    tokenizer = get_tokenizer('basic_english')
    
    # 構建詞彙表
    def yield_tokens(texts):
        for text in texts:
            yield tokenizer(text)
    
    print("Building vocabulary...")
    vocab = build_vocab_from_iterator(
        yield_tokens(train_texts),
        min_freq=5,
        specials=['<unk>', '<pad>']
    )
    vocab.set_default_index(vocab['<unk>'])
    
    return (train_texts, train_labels), (test_texts, test_labels), vocab, tokenizer

def main():
    # 設置參數
    d_model = 128
    nhead = 8
    num_layers = 2
    dim_feedforward = 512
    num_classes = 4  # AG_NEWS有4個類別
    max_len = 128   # 新聞文本可能較長
    batch_size = 32
    
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加載數據
    print("Loading AG_NEWS dataset...")
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
    
    # 視覺化位置編碼和注意力權重
    plot_positional_encoding(save_path=os.path.join(SAVE_DIR, 'ch8_positional_encoding.png'))
    
    # 創建模型
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes,  # 4個類別
        max_len=max_len
    )
    
    # 創建訓練器和優化器
    trainer = TransformerTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
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
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ch8_transformer_classifier.pth'))
    print("\n模型和圖片已保存到", SAVE_DIR)

if __name__ == '__main__':
    main()
