import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from data import X, Y
from sklearn.model_selection import LeaveOneOut

X_base = torch.tensor(X, dtype=torch.float32)
Y_base = torch.tensor(Y, dtype=torch.float32)

X_list = [X_base]
Y_list = [Y_base]

torch.manual_seed(42)

for _ in range(3):
    X_noise_percent = (torch.rand_like(X_base) * 0.1) - 0.05
    Y_noise_percent = (torch.rand_like(Y_base) * 0.1) - 0.05

    X_noisy = X_base * (1.0 + X_noise_percent)
    Y_noisy = Y_base * (1.0 + Y_noise_percent)

    X_list.append(X_noisy)
    Y_list.append(Y_noisy)

X_noisy_total = torch.cat(X_list, dim=0)
Y_noisy_total = torch.cat(Y_list, dim=0)

X_apriori = torch.tensor([
    [0.0, 0.0],  # нулевая скорость и нулевое давление = нет износа, температура не растет
    [0.0, 0.1],  # нулевая скорость и любое давление = нет износа, температура тоже комнатная
    [0.0, 0.3],
    [0.0, 0.5],
    [0.0, 0.6],
    [60.0, 0.0],  # скорость есть, а давления нет = трения нет, износа не будет
    [120.0, 0.0],
    [180.0, 0.0],
    [200.0, 0.0]
], dtype=torch.float32)

Y_apriori = torch.tensor([
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0],
    [0.0, 23.0]
], dtype=torch.float32)

X_tensor = torch.cat([X_noisy_total, X_apriori], dim=0)
Y_tensor = torch.cat([Y_noisy_total, Y_apriori], dim=0)

X_normal = torch.tensor([[0.0, 0.0], [200.0, 0.6]])
Y_normal = torch.tensor([[0.0, 24.0], [15.0, 100.0]])

min_val_P_V = X_normal.min(dim=0).values
min_val_WR_T = Y_normal.min(dim=0).values

max_val_P_V = X_normal.max(dim=0).values
max_val_WR_T = Y_normal.max(dim=0).values

X_norm = ((X_tensor - min_val_P_V) / (max_val_P_V - min_val_P_V)) * 2 - 1
Y_norm = ((Y_tensor - min_val_WR_T) / (max_val_WR_T - min_val_WR_T)) * 2 - 1


class FrictionMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 2)
        self.activation = nn.Tanh()

    def forward(self, x):
        layer1_res = self.hidden(x)
        res = self.activation(layer1_res)
        layer2_res = self.output(res)
        final_res = self.activation(layer2_res)  
        return final_res  


def train(model, optimizer, X, Y, criterion, epochs):
    loss_history = []

    for epoch in range(epochs):
        if isinstance(optimizer, torch.optim.LBFGS):
            def closure():
                optimizer.zero_grad()
                predictions = model(X)
                loss = criterion(predictions, Y)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
            loss_history.append(loss.item())
        else:
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

torch.manual_seed(42)
model_lbfgs = FrictionMLP() 

criterion = nn.MSELoss()
optimizer_adam = torch.optim.Adam(model_adam.parameters(), lr=0.01)
optimizer_sgd = torch.optim.SGD(model_sgd.parameters(), lr=0.01)

optimizer_lbfgs = torch.optim.LBFGS(model_lbfgs.parameters(), lr=0.1, line_search_fn='strong_wolfe')

base_epochs = 5000
lbfgs_epochs = 20 

loss_history_adam = train(model_adam, optimizer_adam, X_norm, Y_norm, criterion, base_epochs)
loss_history_sgd = train(model_sgd, optimizer_sgd, X_norm, Y_norm, criterion, base_epochs)
loss_history_lbfgs = train(model_lbfgs, optimizer_lbfgs, X_norm, Y_norm, criterion, lbfgs_epochs)


def run_loocv(optimizer_class, lr, base_epochs):
    loo = LeaveOneOut()
    fold_losses = []

    actual_epochs = 30 if optimizer_class == torch.optim.LBFGS else base_epochs

    for train_idx, val_idx in loo.split(X_norm):
        train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
        val_idx_tensor = torch.tensor(val_idx, dtype=torch.long)

        X_train, X_val = X_norm[train_idx_tensor], X_norm[val_idx_tensor]
        Y_train, Y_val = Y_norm[train_idx_tensor], Y_norm[val_idx_tensor]

        torch.manual_seed(42)  
        model = FrictionMLP()

        if optimizer_class == torch.optim.LBFGS:
            optimizer = optimizer_class(model.parameters(), lr=lr, line_search_fn='strong_wolfe')
        else:
            optimizer = optimizer_class(model.parameters(), lr=lr)

        train(model, optimizer, X_train, Y_train, criterion, actual_epochs)

        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, Y_val)

        fold_losses.append(val_loss.item())

    return fold_losses

losses_adam = run_loocv(torch.optim.Adam, lr=0.01, base_epochs=500)
losses_sgd = run_loocv(torch.optim.SGD, lr=0.01, base_epochs=500)
losses_lbfgs = run_loocv(torch.optim.LBFGS, lr=0.1, base_epochs=500)  

def predict_single(model, P, V):  
    model.eval()
    test_in = torch.tensor([[P, V]], dtype=torch.float32)
    test_in_norm = ((test_in - min_val_P_V) / (max_val_P_V - min_val_P_V)) * 2 - 1

    with torch.no_grad():
        res_norm = model(test_in_norm)

    res = ((res_norm + 1) / 2) * (max_val_WR_T - min_val_WR_T) + min_val_WR_T
    return res.numpy()

print("Прогноз Adam для [120, 0.2]:", predict_single(model_adam, 120, 0.2))
print("Прогноз SGD для [120, 0.2]:", predict_single(model_sgd, 120, 0.2))
print("Прогноз LBFGS для [120, 0.2]:", predict_single(model_lbfgs, 120, 0.2))

print("Adam mean val loss:", sum(losses_adam) / len(losses_adam))
print("SGD mean val loss:", sum(losses_sgd) / len(losses_sgd))
print("LBFGS mean val loss:", sum(losses_lbfgs) / len(losses_lbfgs))


V_range = np.linspace(0.0, 0.6, 100)
P_range = np.linspace(0.0, 200.0, 100)
P_grid, V_grid = np.meshgrid(P_range, V_range)

grid_points = np.column_stack([P_grid.ravel(), V_grid.ravel()])
grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

grid_norm = (grid_tensor - min_val_P_V) / (max_val_P_V - min_val_P_V) * 2 - 1

def predict_grid(model):
    model.eval()  
    with torch.no_grad():
        pred_norm = model(grid_norm)

    pred = ((pred_norm + 1) / 2) * (max_val_WR_T - min_val_WR_T) + min_val_WR_T

    WR_pred = pred[:, 0].numpy().reshape(P_grid.shape)
    T_pred = pred[:, 1].numpy().reshape(P_grid.shape)
    return WR_pred, T_pred

WR_adam, T_adam = predict_grid(model_adam)
WR_sgd, T_sgd = predict_grid(model_sgd)
WR_lbfgs, T_lbfgs = predict_grid(model_lbfgs)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
models_data = zip(axes, [T_adam, T_sgd, T_lbfgs], [WR_adam, WR_sgd, WR_lbfgs], ['Adam', 'SGD', 'LBFGS (LM-style)'])

levels_T = np.linspace(20.0, 130.0, 23)

for ax, T_pred, WR_pred, title in models_data:
    cf = ax.contourf(P_grid, V_grid, T_pred, levels=levels_T, cmap='inferno', extend='both')
    cl = ax.contour(P_grid, V_grid, WR_pred, levels=8, colors='white', linewidths=0.7)
    ax.clabel(cl, inline=True, fontsize=8, colors='white')

    ax.scatter(X_tensor[:, 0], X_tensor[:, 1],
               c='cyan', edgecolors='black', s=30, zorder=5, label='реальные эксперименты')

    ax.set_xlim(0, 200)
    ax.set_ylim(0, 0.6)

    ax.set_xlabel('Load, N')
    ax.set_ylabel('Velocity, m/s')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=8)
    fig.colorbar(cf, ax=ax, label='Temp, C')

plt.tight_layout()
plt.savefig('combined_contour.png', dpi=150)
plt.show()


P_fixed = 120.0
V_slice = np.linspace(0.0, 0.6, 100)  

slice_points = np.column_stack([np.full_like(V_slice, P_fixed), V_slice])
slice_tensor = torch.tensor(slice_points, dtype=torch.float32)

slice_norm = ((slice_tensor - min_val_P_V) / (max_val_P_V - min_val_P_V)) * 2 - 1

def predict_slice(model):
    model.eval()
    with torch.no_grad():
        pred_norm = model(slice_norm)
    pred = ((pred_norm + 1) / 2) * (max_val_WR_T - min_val_WR_T) + min_val_WR_T
    return pred[:, 0].numpy(), pred[:, 1].numpy()

WR_slice_adam, T_slice_adam = predict_slice(model_adam)
WR_slice_sgd, T_slice_sgd = predict_slice(model_sgd)
WR_slice_lbfgs, T_slice_lbfgs = predict_slice(model_lbfgs)

V_real_exact = np.array([0.1, 0.2, 0.3, 0.5])
WR_real_exact = np.array([8.66348444, 1.8, 1.81686222, 1.62854222])
T_real_exact = np.array([34.0, 38.0, 43.0, 52.0])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(V_slice, WR_slice_adam, 'r-', linewidth=2, label='Adam (прогноз)')
ax1.plot(V_slice, WR_slice_sgd, 'b--', linewidth=2, label='SGD (прогноз)')
ax1.plot(V_slice, WR_slice_lbfgs, 'g-.', linewidth=2, label='LBFGS (прогноз)') 
ax1.scatter(V_real_exact, WR_real_exact, 
            c='cyan', edgecolors='black', s=80, zorder=5, 
            label='4 эксп. точки (P=120)')
ax1.set_xlabel('Velocity (Скорость), m/s')
ax1.set_ylabel('Wear Rate (Износ)')
ax1.set_title(f'Зависимость износа от скорости при P = {P_fixed}')
ax1.grid(True, linestyle=':', alpha=0.6)
ax1.legend()

ax2.plot(V_slice, T_slice_adam, 'r-', linewidth=2, label='Adam (прогноз)')
ax2.plot(V_slice, T_slice_sgd, 'b--', linewidth=2, label='SGD (прогноз)')
ax2.plot(V_slice, T_slice_lbfgs, 'g-.', linewidth=2, label='LBFGS (прогноз)') 
ax2.scatter(V_real_exact, T_real_exact, 
            c='cyan', edgecolors='black', s=80, zorder=5, 
            label='4 эксп. точки (P=120)')
ax2.set_xlabel('Velocity (Скорость), m/s')
ax2.set_ylabel('Temperature (Температура), °C')
ax2.set_title(f'Зависимость температуры от скорости при P = {P_fixed}')
ax2.grid(True, linestyle=':', alpha=0.6)
ax2.legend()

ax1.set_xlim(0.0, 0.6)
ax1.set_ylim(0.0, 12.0)

ax2.set_xlim(0.0, 0.6)
ax2.set_ylim(20.0, 130.0)

plt.tight_layout()
plt.savefig('frictional_slice_120.png', dpi=150)
plt.show()
