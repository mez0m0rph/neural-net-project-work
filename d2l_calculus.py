import torch

x = torch.arange(4.0)
y = torch.arange(4.0)

# [requires_grad_] - это метка для PyTorch, что этот тензор - веса нашей модели. Начни записывать все изменения (для производной)
x.requires_grad_(True)
x.grad

y = 2 * torch.dot(x, y)
print(y)