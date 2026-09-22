import math
import numpy as np
import matplotlib.pyplot as plt

# гауссовское распределение
# f(x) = (1 / (σ * sqrt(2π))) * exp(-((x - μ)^2) / (2 * σ^2))

def normal(x, mu, sigma):
    # mu - среднее значение (мат ожидание) - центра распределения (вершина колокола)
    # mu - координата по оси x 
    # sigma - стандартное (среднеквадратичное отклонение) - ширина колокола
    # mu - расстояние по оси x
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 * (x - mu)**2 / sigma ** 2)

params = [(0, 1), (0, 2), (3, 1)]  # набор [ожидание (центр) + отклонение]
x = np.arange(-7, 7, 0.01)

plt.figure(figsize=(7, 4))  # окно для графика 7x4 дюймак
for mu, sigma in params:
    y = [normal(xi, mu, sigma) for xi in x]
    plt.plot(x, y, label=f'mu={mu}, sigma={sigma}')

plt.xlabel('x')
plt.ylabel('p(x)')
plt.legend()
plt.grid(True)
plt.show()