import numpy as np
import matplotlib.pyplot as plt

np.random.seed(56)

X = np.linspace(-3, 3, 10).reshape(-1, 1)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, size=10)

K = 10  # число нейронов на скрытом RBF-слое

# выбор случайных точек в качестве центроидов (БЕЗ ДУБЛИКАТОВ!)
idx = np.random.choice(len(X), size=K, replace=False)
centers = X[idx]  # форма (K, 1) - c_1..c_K
sigma = 1.0  # ширина гауссианы (пока одна для всех)

# dists_sq[i, j] = ‖x_i − c_j‖² = Σ_d (x_i[d] − c_j[d])²
# квадрат евклидова расстояния между всеми парами точек X и центров centers
dists_sq = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # суммирует по последней оси (по признакам/координатам)
# радиально-базисная функция (Гауссина)
Phi = np.exp(-dists_sq / (2 * sigma**2))  # превращает расстояние в "оценку похожести" (матрица активаций)

Phi_b = np.hstack([Phi, np.zeros((Phi.shape[0], 1))])  # столбец единиц под смещение b (приклеивается)
w, *_ = np.linalg.lstsq(Phi_b, y, rcond=None)  # Phi_b @ w ≈ y
weights, bias = w[:-1], w[-1]

def predict(X_new):
    d2 = ((X_new[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # считаем квадрат расстояния
    Phi_new = np.exp(-d2 / (2 * sigma**2))  # превращаем расстояния в отклики нейронов
    return Phi_new @ weights + bias  # итоговое предсказание

# новые точки (их не было в обучении)
X_test = np.linspace(-3, 3, 200).reshape(-1, 1)
X_test2 = np.linspace(-10, 10, 200).reshape(-1, 1)
y_pred = predict(X_test2)


plt.scatter(X, y, s=15, color="gray", label="обучающие точки (с шумом)")
plt.plot(X_test, np.sin(X_test), "--", color="black", label="истинная sin(x)")
plt.plot(X_test2, y_pred, color="red", label="предсказание RBF-сети")
plt.scatter(centers, np.zeros_like(centers), marker="^", color="blue", label="центры RBF")
plt.legend()
plt.show()