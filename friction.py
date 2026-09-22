import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data import X, Y
from sklearn.model_selection import LeaveOneOut

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

X_mean = X_tensor.mean(dim=0)
X_std = X_tensor.std(dim=0, correction=0)

X_normal = torch.tensor([[0, 0], [200, 0.6]])
Y_normal = torch.tensor([[0, 24], [15, 100]])

min_val_P_V = X_normal.min(dim=0).values
min_val_WR_T = Y_normal.min(dim=0).values

max_val_P_V = X_normal.max(dim=0).values
max_val_WR_T = Y_normal.max(dim=0).values

X_norm = ((X_tensor - min_val_P_V) / (max_val_P_V - min_val_P_V))
X_norm1 = X_norm * (max_val_P_V - min_val_P_V) + min_val_P_V

print(X_norm)
print(X_norm1)

Y_mean = Y_tensor.mean(dim=0)
Y_std = Y_tensor.std(dim=0, correction=0)

Y_norm = ((Y_tensor - min_val_WR_T) / (max_val_WR_T - min_val_WR_T))


class FrictionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 17)
        self.output = nn.Linear(17, 2)
        self.activation = nn.ReLU()

    def forward(self, x):
        layer1_res = self.hidden(x)
        res = self.activation(layer1_res)
        layer2_res = self.output(res)
        return layer2_res
    

def train(model, optimizer, X, Y, criterion, epochs):
        loss_history = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X)
            loss = criterion(predictions, Y)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
        return loss_history

torch.manual_seed(42)
model_adam = FrictionMLP()

torch.manual_seed(42)
model_sgd = FrictionMLP()

criterion = nn.MSELoss()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)

epochs = 500

loss_history_adam = train(model_adam, optimizer_adam, X_norm, Y_norm, criterion, epochs)
loss_history_sgd = train(model_sgd, optimizer_sgd, X_norm, Y_norm, criterion, epochs)



def run_loocv(optimizer_class, lr, epochs):
    loo = LeaveOneOut()
    fold_losses = []

    for train_idx, val_idx in loo.split(X_norm):
        X_train, X_val = X_norm[train_idx], X_norm[val_idx]
        Y_train, Y_val = Y_norm[train_idx], Y_norm[val_idx]

        model = FrictionMLP()
        optimizer = optimizer_class(model.parameters(), lr=lr)

        train(model, optimizer, X_train, Y_train, criterion, epochs)

        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val)

        fold_losses.append(val_loss.item())

    return fold_losses

losses_adam = run_loocv(torch.optim.Adam, lr=0.05, epochs=500)
losses_sgd = run_loocv(torch.optim.SGD, lr=0.05, epochs=500)

print("Adam mean val loss:", sum(losses_adam) / len(losses_adam))
print("SGD mean val loss:", sum(losses_sgd) / len(losses_sgd))

P_range = np.linspace(X_tensor[:, 0].min(), X_tensor[:, 0].max(), 100)
V_range = np.linspace(X_tensor[:, 1].min(), X_tensor[:, 1].max(), 100)
P_grid, V_grid = np.meshgrid(P_range, V_range)

grid_points = np.column_stack([P_grid.ravel(), V_grid.ravel()])
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)
grid_norm = (grid_tensor - min_val_P_V) / (max_val_P_V - min_val_P_V)

def predict_grid(model):
    with torch.no_grad():
        pred_norm = model(grid_norm)
    pred = pred_norm * (max_val_WR_T - min_val_WR_T) + min_val_WR_T
    WR_pred = pred[:, 0].numpy().reshape(P_grid.shape)
    T_pred = pred[:, 1].numpy().reshape(P_grid.shape)
    return WR_pred, T_pred

WR_adam, T_adam = predict_grid(model_adam)
WR_sgd, T_sgd = predict_grid(model_sgd)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, T_pred, WR_pred, title in zip(axes, [T_adam, T_sgd], [WR_adam, WR_sgd], ['Adam', 'SGD']):
    cf = ax.contourf(P_grid, V_grid, T_pred, levels=20, cmap='inferno')
    cl = ax.contour(P_grid, V_grid, WR_pred, levels=8, colors='white', linewidths=0.7)
    ax.clabel(cl, inline=True, fontsize=8, colors='white')

    ax.scatter(X_tensor[:, 0], X_tensor[:, 1],
               c='cyan', edgecolors='black', s=30, zorder=5, label='реальные эксперименты')

    ax.set_xlabel('Load, N')
    ax.set_ylabel('Velocity, m/s')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(cf, ax=ax, label='Temp, C')

plt.tight_layout()
plt.savefig('combined_contour.png', dpi=150)
plt.show()
