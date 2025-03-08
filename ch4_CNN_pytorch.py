"""
PyTorch CNN 實作教學
=================

本章節介紹卷積神經網路(CNN)的實現，包括：
1. CNN 基本組件介紹
2. 經典 CNN 架構實作
3. MNIST 手寫數字識別實戰
4. 模型訓練與評估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class SimpleCNN(nn.Module):
    """簡單的CNN模型"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一個卷積層
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # 輸入: 28x28 -> 26x26
        # 第二個卷積層
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)  # 13x13 -> 11x11
        # 全連接層
        self.fc1 = nn.Linear(64 * 5 * 5, 128)  # 5x5 是經過兩次池化後的大小
        self.fc2 = nn.Linear(128, 10)
        
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 第一個卷積層 + ReLU + MaxPool
        x = F.relu(self.conv1(x))        # 28x28 -> 26x26
        x = F.max_pool2d(x, 2)           # 26x26 -> 13x13
        
        # 第二個卷積層 + ReLU + MaxPool
        x = F.relu(self.conv2(x))        # 13x13 -> 11x11
        x = F.max_pool2d(x, 2)           # 11x11 -> 5x5
        
        # 展平
        x = x.view(-1, 64 * 5 * 5)       # 64個通道，每個5x5
        
        # 全連接層
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

class CNNTrainer:
    """CNN訓練器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.train_accs = []
        self.test_losses = []
        self.test_accs = []

    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        running_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        self.train_losses.append(running_loss/len(train_loader))
        self.train_accs.append(100.*correct/total)
        return running_loss/len(train_loader), 100.*correct/total

    def test(self, test_loader, criterion):
        self.model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        test_loss /= len(test_loader)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        self.test_losses.append(test_loss)
        self.test_accs.append(accuracy)
        return test_loss, accuracy

    def plot_training_history(self):
        """繪製訓練歷史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.test_accs, label='Test Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'ch4_cnn_training_history.png'))
        plt.close()

def visualize_predictions(model, test_loader, device, num_images=10):
    """視覺化預測結果"""
    model.eval()
    images, labels = next(iter(test_loader))
    images = images[:num_images].to(device)
    labels = labels[:num_images]
    
    with torch.no_grad():
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
    
    fig = plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i].cpu().squeeze(), cmap='gray')
        color = 'green' if predictions[i] == labels[i] else 'red'
        plt.title(f'Pred: {predictions[i]}\nTrue: {labels[i]}', color=color)
        plt.axis('off')
    
    plt.suptitle('CNN Predictions')  # 改為英文
    plt.savefig(os.path.join(SAVE_DIR, 'ch4_cnn_predictions.png'))
    plt.close()

def plot_feature_maps(model, test_loader, device, layer_name='conv1'):
    """視覺化特徵圖"""
    model.eval()
    images, _ = next(iter(test_loader))
    image = images[0:1].to(device)
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    if hasattr(model, layer_name):
        getattr(model, layer_name).register_forward_hook(get_activation(layer_name))
    
    model(image)
    act = activation[layer_name].squeeze().cpu()
    
    fig = plt.figure(figsize=(12, 8))
    for i in range(min(32, act.size(0))):
        plt.subplot(4, 8, i+1)
        plt.imshow(act[i], cmap='viridis')
        plt.axis('off')
    plt.suptitle(f'Feature Maps of {layer_name} Layer')  # 改為英文
    
    plt.savefig(os.path.join(SAVE_DIR, f'ch4_cnn_feature_maps_{layer_name}.png'))
    plt.close()

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 數據轉換
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加載MNIST數據集
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)
    
    # 創建模型和訓練器
    model = SimpleCNN()
    trainer = CNNTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        test_loss, test_acc = trainer.test(test_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # 繪製訓練歷史
    trainer.plot_training_history()
    
    # 視覺化預測結果
    visualize_predictions(model, test_loader, device)
    
    # 視覺化特徵圖
    plot_feature_maps(model, test_loader, device, 'conv1')
    plot_feature_maps(model, test_loader, device, 'conv2')
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ch4_mnist_cnn.pth'))
    print("\n模型和圖片已保存到", SAVE_DIR)

if __name__ == '__main__':
    main()
