"""
PyTorch 自定義神經網路
===================

本章節介紹如何創建自定義神經網路，包括：
1. 基本神經網路組件
2. 自定義層的實現
3. 複雜模型架構的設計
4. 模型參數的管理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class CustomLayer(nn.Module):
    """自定義層示例"""
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x):
        # 自定義前向傳播邏輯
        out = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            out += self.bias
        return out

class ResidualBlock(nn.Module):
    """殘差塊示例"""
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ComplexNet(nn.Module):
    """複雜網絡架構示例"""
    def __init__(self, num_classes=10):
        super(ComplexNet, self).__init__()
        
        # 特徵提取部分
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            ResidualBlock(128)
        )
        
        # 分類部分
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )
        
        # 初始化權重
        self._initialize_weights()
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def visualize_model_structure(model):
    """視覺化模型結構"""
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\nModel Structure:")
    print("="*50)
    print(model)
    print("\nTotal trainable parameters:", count_parameters(model))
    
    # 打印每層參數數量
    print("\nParameters per layer:")
    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()}")

def plot_model_architecture(model, save_path):
    """視覺化模型架構"""
    plt.figure(figsize=(12, 8))
    layers = [name for name, _ in model.named_modules() if len(name) > 0]
    
    for i, layer in enumerate(layers):
        plt.plot([0, 1], [i, i], 'b-', linewidth=2)
        plt.text(1.1, i, layer, fontsize=10)
    
    plt.title('Model Architecture')
    plt.xlabel('Layer Depth')
    plt.ylabel('Layer Name')
    plt.grid(True)
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def plot_parameter_distribution(model, save_path):
    """視覺化模型參數分佈"""
    plt.figure(figsize=(12, 4))
    
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.requires_grad:
            plt.subplot(1, 2, 1)
            plt.hist(param.data.cpu().numpy().flatten(), bins=50, alpha=0.5, label=name)
            plt.title('Parameter Distribution')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # 創建模型實例
    model = ComplexNet()
    
    # 視覺化模型結構
    visualize_model_structure(model)
    
    # 測試前向傳播
    print("\nTesting forward pass:")
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # 測試自定義層
    custom_layer = CustomLayer(10, 5)
    x = torch.randn(3, 10)
    output = custom_layer(x)
    print("\nTesting custom layer:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    main()
