"""
PyTorch 資料處理基礎
==================

本章節介紹 PyTorch 中的資料處理機制，包括：
1. Dataset 類別的創建和使用
2. DataLoader 的配置和應用
3. 資料轉換和預處理
4. 批次處理示例
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data  # 現在data將是直接的numpy數組
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 直接使用numpy數組而不是讀取文件
        image = self.data[idx]
        image = Image.fromarray((image * 255).astype('uint8'))
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def demonstrate_transforms(save_path):
    """展示數據轉換"""
    image = np.random.rand(100, 100, 3)
    pil_image = Image.fromarray((image * 255).astype('uint8'))
    
    transforms_list = {
        'Original': None,
        'Resize': transforms.Resize((50, 50)),
        'RandomCrop': transforms.RandomCrop(80),
        'ColorJitter': transforms.ColorJitter(brightness=0.5),
        'RandomRotation': transforms.RandomRotation(45)
    }
    
    plt.figure(figsize=(15, 3))
    for i, (name, transform) in enumerate(transforms_list.items()):
        plt.subplot(1, 5, i+1)
        if transform:
            img_transformed = transform(pil_image)
        else:
            img_transformed = pil_image
        plt.imshow(img_transformed)
        plt.title(name)
        plt.axis('off')
    
    plt.suptitle('Data Transformation Examples')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def visualize_batch(dataloader, save_path):
    """視覺化批次數據"""
    images, labels = next(iter(dataloader))
    
    batch_size = images.size(0)
    grid_size = int(np.ceil(np.sqrt(batch_size)))
    
    plt.figure(figsize=(10, 10))
    for i in range(min(batch_size, 16)):
        plt.subplot(4, 4, i+1)
        plt.imshow(images[i].numpy().transpose(1, 2, 0))
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    
    plt.suptitle('Batch Data Visualization')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_data_distribution(dataset, save_path):
    """繪製數據分佈"""
    labels = [label.item() for _, label in dataset]
    
    plt.figure(figsize=(8, 4))
    plt.hist(labels, bins=len(set(labels)), rwidth=0.8)
    plt.title('Dataset Label Distribution')
    plt.xlabel('Label')
    plt.ylabel('Count')
    
    plt.savefig(save_path)
    plt.close()

def main():
    # 生成隨機圖像數據
    num_images = 10
    image_size = 128
    data = np.random.rand(num_images, image_size, image_size, 3)  # 生成隨機RGB圖像
    labels = np.random.randint(0, 2, num_images)  # 生成隨機標籤

    # 數據轉換
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # 創建數據集和數據加載器
    dataset = CustomDataset(data, labels, transform=transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 視覺化示例
    demonstrate_transforms(os.path.join(SAVE_DIR, 'ch1_data_transforms.png'))
    visualize_batch(dataloader, os.path.join(SAVE_DIR, 'ch1_batch_visualization.png'))
    plot_data_distribution(dataset, os.path.join(SAVE_DIR, 'ch1_data_distribution.png'))

    print("\n所有圖片已保存到", SAVE_DIR)

if __name__ == '__main__':
    main() 