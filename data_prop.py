import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

X_raw = np.array([
    [0.0, 0.0],
    [10.0, 0.2],
    [20.0, 0.2],
    [30.0, 0.2],
    [40.0, 0.2],
    [50.0, 0.2],
    [10.0, 0.5],
    [20.0, 0.5],
    [30.0, 0.5],
    [10.0, 1.0],
    [20.0, 1.0],
    [30.0, 1.0],
    [10.0, 3.0],
    [20.0, 3.0],
    [30.0, 3.0]
], dtype=np.float32)

Y_raw = np.array([
    [4.2, 101.6, 26.16, 65.1],
    [5.12, 89.8, 2.93, 65.2],
    [6.05, 85.7, 2.13, 67.1],
    [9.68, 99.1, 1.56, 83.6],
    [17.42, 153.5, 1.86, 118.3],
    [23.15, 165.9, 1.46, 132.8 ],
    [6.48, 98.9, 2.43, 75.8],
    [11.35, 118.3, 1.44, 102.1],
    [16.67, 157.7, 1.56, 125.4],
    [6.18, 104.7, 3.17, 77.7],
    [9.82, 128.4, 2.32, 98.5],
    [15.25, 153.2, 1.68, 127.4],
    [7.72, 116.3, 2.08, 104.8],
    [11.86, 135.4, 1.49, 124.9],
    [18.43, 171.4, 1.31, 154.8]
], dtype=np.float32)

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

# нормализация в диапазон [0, 1]
X_scaled = scaler_X.fit_transform(X_raw)
Y_scaled = scaler_Y.fit_transform(Y_raw)

# генерация виртуальных точек для контроля физики
def get_physics_point(n_points=1000):
    cf_grid = np.linspace(0.0, 50.0, int(np.sqrt(n_points)))
    lenght_grid = np.linspace(0.0, 3.0, int(np.sqrt(n_points)))

    cf_mesh, lenght_mesh = np.meshgrid(cf_grid, lenght_grid)
    X_phys_raw = np.vstack([cf_mesh.ravel(), lenght_mesh.ravel()]).T.astype(np.float32)

    X_phys_scaled = scaler_X.transform(X_phys_raw)
    return torch.tensor(X_phys_scaled, dtype=torch.float32)

# device = "cuda" для проведения вычислений на видеокарте
def get_pytorch_data(device):
    x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(Y_scaled, dtype=torch.float32).to(device)
    return x_tensor, y_tensor 