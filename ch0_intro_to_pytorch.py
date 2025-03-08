"""
PyTorch 基礎入門
===============

本章節將介紹 PyTorch 的基礎概念和操作，包括：
1. 張量(Tensor)的創建和操作
2. 基本數學運算
3. 自動微分(Autograd)機制
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 確保圖片保存目錄存在
SAVE_DIR = 'figures'
os.makedirs(SAVE_DIR, exist_ok=True)

def tensor_basics():
    print("1. 創建張量的多種方式:")
    # 從列表創建
    tensor1 = torch.tensor([[1, 2], [3, 4]])
    print("從列表創建:\n", tensor1)
    
    # 從NumPy數組創建
    numpy_array = np.array([[1, 2], [3, 4]])
    tensor2 = torch.from_numpy(numpy_array)
    print("\n從NumPy創建:\n", tensor2)
    
    # 創建特殊張量
    print("\n特殊張量:")
    print("零張量:\n", torch.zeros(2, 3))
    print("單位張量:\n", torch.ones(2, 3))
    print("隨機張量:\n", torch.rand(2, 3))
    print("對角張量:\n", torch.eye(3))

def tensor_operations():
    print("\n2. 張量運算示例:")
    x = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
    y = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)
    
    print("加法:\n", x + y)
    print("\n乘法:\n", x * y)  # 元素級乘法
    print("\n矩陣乘法:\n", torch.matmul(x, y))
    
    # 張量維度操作
    print("\n改變形狀:")
    print("原始張量:\n", x)
    print("重塑後:\n", x.reshape(1, 4))
    print("轉置後:\n", x.t())

def autograd_example():
    print("\n3. 自動微分示例:")
    # 創建需要計算梯度的張量
    x = torch.tensor([[1., 2.], [3., 4.]], requires_grad=True)
    print("輸入張量 x:\n", x)
    
    # 進行一些運算
    y = x + 2
    z = y * y * 2
    out = z.mean()
    
    print("計算過程:")
    print("y = x + 2")
    print("z = y * y * 2")
    print("out = z.mean()")
    
    # 計算梯度
    out.backward()
    print("\n計算得到的梯度:\n", x.grad)

def visualize_tensor(save_path):
    """視覺化張量"""
    tensor = torch.rand(10, 10)
    
    plt.figure(figsize=(6, 6))
    plt.imshow(tensor, cmap='viridis')
    plt.colorbar()
    plt.title("2D Tensor Visualization")
    
    plt.savefig(save_path)
    plt.close()

def plot_computation_graph(save_path):
    """繪製計算圖示例"""
    plt.figure(figsize=(10, 6))
    
    # 繪製節點
    plt.plot([0, 1, 2], [0, 0, 0], 'bo', markersize=15, label='Tensors')
    plt.text(-0.1, 0.1, 'x', fontsize=12)
    plt.text(0.9, 0.1, 'y', fontsize=12)
    plt.text(1.9, 0.1, 'z', fontsize=12)
    
    # 繪製箭頭
    plt.arrow(0.1, 0, 0.8, 0, head_width=0.1, head_length=0.1)
    plt.arrow(1.1, 0, 0.8, 0, head_width=0.1, head_length=0.1)
    
    # 添加運算說明
    plt.text(0.4, 0.2, '*2', fontsize=12)
    plt.text(1.4, 0.2, '*y + 1', fontsize=12)
    
    plt.title('PyTorch Computation Graph Example')
    plt.axis('off')
    
    plt.savefig(save_path)
    plt.close()

def plot_gradient_flow(save_path):
    """視覺化梯度流動"""
    x = torch.linspace(-5, 5, 100, requires_grad=True)
    y = torch.sigmoid(x)
    y.sum().backward()
    
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x.detach(), y.detach(), label='Sigmoid')
    plt.title('Sigmoid Function')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x.detach(), x.grad, label='Gradient')
    plt.title('Gradient Values')
    plt.grid(True)
    plt.legend()
    
    plt.savefig(save_path)
    plt.close()

def main():
    print("="*50)
    print("PyTorch版本:", torch.__version__)
    print("="*50)
    
    # 基本操作示例
    tensor_basics()
    tensor_operations()
    autograd_example()
    
    # 視覺化示例
    visualize_tensor(os.path.join(SAVE_DIR, 'ch0_tensor_visualization.png'))
    plot_computation_graph(os.path.join(SAVE_DIR, 'ch0_computation_graph.png'))
    plot_gradient_flow(os.path.join(SAVE_DIR, 'ch0_gradient_flow.png'))
    
    print("\n所有圖片已保存到", SAVE_DIR)

if __name__ == "__main__":
    main()


