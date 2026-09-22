import torch
import torch.nn as nn

class LinearRegressionScratch(nn.Module):
    def __init__(self, num_inputs, lr, wd=0.01, sigma = 0.01):
        super().__init__()

        self.lr = lr

        self.wd = wd

        # инициализирует тензор, заполненный случайными маленькими числами
        self.w = torch.normal(0, sigma, (num_inputs, 1), requires_grad=True)

        # (1, ...) - 1 это размерность тензора [вектор]
        # (..., requires_grad=True) - отслеживает все операции и считает производные
        self.b = torch.zeros(1, requires_grad=True)

    def forward(self, X):
        return torch.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):  # среднеквадратичная ошибка  
        y_safe = y.reshape(y_hat.shape)
        l = (y_hat - y_safe) ** 2 / 2
        return l.mean()

    def configure_optimizers(self):
        return SGD([self.w, self.b], self.lr, self.wd)  # создает экземпляр класса с параметрами

class SGD:  # стохастический градиентный спуск
    def __init__(self, params, lr, wd=0.0):
        # params - это список тензоров, которые мы хотим обучить
        self.params = params

        self.lr = lr

        self.wd = wd

    def step(self):
        for param in self.params:
            if param.ndim > 1:
                # математика штраф - добавляем текущее значение веса к его градиенту
                param.grad.data += self.wd * param.data

            param.data -= self.lr * param.grad.data

    def zero_grad(self):  # обнуление градиентов
        for param in self.params:
            if param.grad is not None:
                param.grad.zero_()