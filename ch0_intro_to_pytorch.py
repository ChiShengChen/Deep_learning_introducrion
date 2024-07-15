## Introduction to PyTorch
# Installing PyTorch:
# 1. going to https://pytorch.org/ and click "Get Started"
# 2. sellect the OS, version of yours env (anaconda recommanded)
# 3. copy and run the command, for example: 
#    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

## Basic Tensor Operations:
import torch

# Create a tensor
x = torch.tensor([[1, 2], [3, 4]])
print(x)

# Basic operations
y = torch.ones_like(x)
print(y)

z = x + y
print(z)

# Matrix multiplication
result = torch.matmul(x, y.T)
print(result)


## Tensor Creation and Operations:
# Create tensors
a = torch.zeros((2, 2))
b = torch.ones((2, 2))
c = torch.rand((2, 2))

print("Zero Tensor:\n", a)
print("Ones Tensor:\n", b)
print("Random Tensor:\n", c)

# Basic arithmetic operations
d = a + b
e = b * c
f = torch.exp(c)

print("Addition:\n", d)
print("Element-wise multiplication:\n", e)
print("Exponent:\n", f)


## Automatic Differentiation with Autograd:
x = torch.ones(2, 2, requires_grad=True)
y = x + 2
z = y * y * 3

# Compute gradients
z = z.mean()
z.backward()

print("Gradients:\n", x.grad)


