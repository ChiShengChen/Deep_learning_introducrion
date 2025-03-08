"""
PyTorch 模型訓練基礎
=================

本章節介紹完整的模型訓練流程，包括：
1. 訓練循環的構建
2. 驗證循環的實現
3. 模型評估和保存
4. 訓練過程可視化
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # numpy array
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接使用numpy數組
        image = self.data[idx]
        image = Image.fromarray((image * 255).astype('uint8'))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

class SimpleNet(nn.Module):
    """簡單的神經網絡模型"""
    def __init__(self, input_size, hidden_size, num_classes):
        super(SimpleNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平輸入
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

class TrainingManager:
    """訓練管理器"""
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self, train_loader, optimizer, criterion):
        """執行一個訓練周期"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 使用tqdm顯示進度條
        with tqdm(train_loader, desc="Training") as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # 清零梯度
                optimizer.zero_grad()
                
                # 前向傳播
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # 反向傳播
                loss.backward()
                optimizer.step()
                
                # 統計
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新進度條
                pbar.set_postfix({
                    'loss': f'{running_loss/total:.3f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        self.train_losses.append(epoch_loss)
        self.train_accs.append(epoch_acc)
        return epoch_loss, epoch_acc
    
    def validate(self, val_loader, criterion):
        """執行驗證"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100. * correct / total
        self.val_losses.append(val_loss)
        self.val_accs.append(val_acc)
        return val_loss, val_acc
    
    def plot_training_history(self):
        """繪製訓練歷史"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(SAVE_DIR, 'training_history.png'))
        plt.close()

def visualize_predictions(model, val_loader, device, save_path):
    """視覺化預測結果"""
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.numpy())
    
    plt.figure(figsize=(10, 5))
    plt.scatter(range(len(predictions)), predictions, c=labels, cmap='coolwarm')
    plt.colorbar(label='True Label')
    plt.title('Model Predictions')
    plt.xlabel('Sample Index')
    plt.ylabel('Predicted Class')
    plt.grid(True)
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 設置設備
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 生成隨機數據
    num_images = 100
    image_size = 128
    data = np.random.rand(num_images, image_size, image_size, 3)  # 隨機RGB圖像
    labels = np.random.randint(0, 2, num_images)  # 二分類標籤

    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # 創建數據集
    dataset = CustomDataset(data, labels, transform=transform)
    
    # 分割數據集
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # 創建數據加載器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 創建模型和訓練組件
    model = SimpleNet(input_size=128*128*3, hidden_size=128, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 創建訓練管理器
    trainer = TrainingManager(model, device)
    
    # 訓練模型
    num_epochs = 10
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        val_loss, val_acc = trainer.validate(val_loader, criterion)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 繪製訓練歷史
    trainer.plot_training_history()
    
    # 視覺化預測結果
    visualize_predictions(model, val_loader, device, 
                        os.path.join(SAVE_DIR, 'ch2_prediction_distribution.png'))
    
    # 保存模型
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'ch2_simple_model.pth'))
    print("\n模型和圖片已保存到", SAVE_DIR)

if __name__ == "__main__":
    main()
