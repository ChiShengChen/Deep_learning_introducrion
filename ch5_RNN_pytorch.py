"""
PyTorch RNN 實作教學
=================

本章節介紹循環神經網路(RNN)的實現，包括：
1. RNN 基本原理與架構
2. 文本分類實作
3. 序列數據處理
4. 模型訓練與評估
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
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from collections import Counter
import string

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class TextRNN(nn.Module):
    """基本的RNN文本分類模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def to(self, device):
        self.device = device
        return super().to(device)
    
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, hidden = self.rnn(embedded)
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [1, batch_size, hidden_dim]
        
        hidden = hidden.squeeze(0)
        # hidden shape: [batch_size, hidden_dim]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class SimpleTextDataset(Dataset):
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
        
        # 將文本轉換為索引序列
        indices = [self.vocab.get(word, 0) for word in self.tokenizer(text)]
        
        # 填充或截斷到固定長度
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        
        return torch.tensor(indices), torch.tensor(label)

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

def plot_sequence_length_distribution(texts, save_path):
    """繪製序列長度分布"""
    lengths = [len(text.split()) for text in texts]
    
    plt.figure(figsize=(10, 5))
    plt.hist(lengths, bins=50, alpha=0.7)
    plt.title('Distribution of Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Count')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def plot_word_frequency(texts, top_n=20, save_path=None):
    """繪製詞頻分布"""
    word_freq = {}
    for text in texts:
        for word in text.split():
            word_freq[word] = word_freq.get(word, 0) + 1
    
    sorted_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, freqs = zip(*sorted_freq)
    
    plt.figure(figsize=(12, 5))
    plt.bar(words, freqs)
    plt.title('Word Frequency Distribution')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(save_path)
    plt.close()

class RNNTrainer:
    """RNN訓練器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for texts, labels in pbar:
            texts, labels = texts.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(texts)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        return running_loss/len(train_loader), 100.*correct/total
    
    def evaluate(self, val_loader, criterion):
        self.model.eval()
        running_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts, labels = texts.to(self.device), labels.to(self.device)
                output = self.model(texts)
                loss = criterion(output, labels)
                
                running_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)
        
        return running_loss/len(val_loader), 100.*correct/total

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
        plt.savefig(os.path.join(SAVE_DIR, 'rnn_training_history.png'))
        plt.close()

def visualize_attention_weights(model, text, vocab, save_path):
    """視覺化RNN的注意力權重"""
    model.eval()
    indices = [vocab.get(word, 0) for word in text.split()]
    input_tensor = torch.tensor(indices).unsqueeze(0).to(model.device)
    
    with torch.no_grad():
        output = model(input_tensor)
    
    if hasattr(model, 'attention_weights'):
        weights = model.attention_weights.squeeze().cpu().numpy()
        
        plt.figure(figsize=(10, 4))
        plt.imshow(weights, cmap='viridis')
        plt.colorbar(label='Weight')
        plt.xticks(range(len(text.split())), text.split(), rotation=45)
        plt.ylabel('Attention Weight')
        plt.title('Attention Visualization')
        plt.tight_layout()
        
        plt.savefig(save_path)
        plt.close()

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成示例數據
    (train_texts, train_labels), (test_texts, test_labels), vocab, tokenizer = generate_text_data()
    
    # 繪製序列長度分布
    plot_sequence_length_distribution(train_texts, 
                                   os.path.join(SAVE_DIR, 'ch5_sequence_length_distribution.png'))
    
    # 繪製詞頻分布
    plot_word_frequency(train_texts, top_n=20, 
                       save_path=os.path.join(SAVE_DIR, 'ch5_word_frequency.png'))
    
    # 創建數據集
    train_dataset = SimpleTextDataset(train_texts, train_labels, vocab, tokenizer, max_len=100)
    test_dataset = SimpleTextDataset(test_texts, test_labels, vocab, tokenizer, max_len=100)
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    # 創建模型
    model = TextRNN(
        vocab_size=len(vocab),
        embedding_dim=100,
        hidden_dim=256,
        output_dim=2  # IMDB是二分類
    )
    
    # 創建訓練器
    trainer = RNNTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_acc = trainer.evaluate(test_loader, criterion)
        
        trainer.history['train_loss'].append(train_loss)
        trainer.history['train_acc'].append(train_acc)
        trainer.history['val_loss'].append(val_loss)
        trainer.history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 繪製訓練歷史
    trainer.plot_training_history()
    
    # 視覺化注意力權重（如果模型支持）
    sample_text = " ".join(map(str, range(10)))  # 示例文本
    visualize_attention_weights(model, sample_text, vocab,
                              os.path.join(SAVE_DIR, 'attention_weights.png'))
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'text_rnn.pth'))
    print("\n模型和圖片已保存到", SAVE_DIR)

if __name__ == '__main__':
    main()
