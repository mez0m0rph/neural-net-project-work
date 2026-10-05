import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import prop_data_2_lg as data_prop
import random
import os
import sys

random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

X, Y = data_prop.get_pytorch_data(device)
X = X.to(device)
Y = Y.to(device)

X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()

X_phys = data_prop.get_physics_point(1000).to(device)

loo = LeaveOneOut()

y_true_all = []
y_pred_all = []

class AdvancedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.SiLU(),
            nn.Linear(32, 32),
            nn.SiLU(),
            nn.Linear(32, 4)
        )
        
    def forward(self, x):
        return self.network(x)

total_folds = X_np.shape[0]
print(f"Starting PINN LOOCV validation on {device}... Total iterations: {total_folds}")

for fold, (train_idx, val_idx) in enumerate(loo.split(X_np)):
    x_train_t = X[train_idx].to(device)
    y_train_t = Y[train_idx].to(device)
    x_val_t = X[val_idx].to(device)
    y_val_t = Y[val_idx].to(device)
    
    model = AdvancedPINN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    epochs = 2000
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        predictions = model(x_train_t)
        loss_data = criterion(predictions, y_train_t)
        
        X_phys.requires_grad_(True)
        pred_phys = model(X_phys)
        E_pred = pred_phys[:, 0]
        UTS_pred = pred_phys[:, 1]
        
        grad_E = torch.autograd.grad(
            outputs=E_pred,
            inputs=X_phys,
            grad_outputs=torch.ones_like(E_pred).to(device),
            create_graph=True
        )[0]
        
        grad_UTS = torch.autograd.grad(
            outputs=UTS_pred,
            inputs=X_phys,
            grad_outputs=torch.ones_like(UTS_pred).to(device),
            create_graph=True
        )[0]
        
        dE_dWcf = grad_E[:, 0]
        dE_dlcf = grad_E[:, 1]
        dUTS_dWcf = grad_UTS[:, 0]
        dUTS_dlcf = grad_UTS[:, 1]
        
        loss_physics = (
            torch.mean(torch.relu(-dE_dWcf)) +
            torch.mean(torch.relu(-dE_dlcf)) +
            torch.mean(torch.relu(-dUTS_dWcf)) +
            torch.mean(torch.relu(-dUTS_dlcf))
        )
        
        lambda_p = min(0.5, 0.01 * (epoch / 100))
        total_loss = loss_data + lambda_p * loss_physics
        total_loss.backward()
        optimizer.step()
        
        model.eval()
        with torch.no_grad():
            val_predictions = model(x_val_t)
            val_loss = criterion(val_predictions, y_val_t)
            
        scheduler.step(val_loss)
        
    model.eval()
    with torch.no_grad():
        val_pred = model(x_val_t).cpu().numpy()
        
    y_true_all.append(Y_np[val_idx])
    y_pred_all.append(val_pred)
    
    if (fold + 1) % 5 == 0 or (fold + 1) == total_folds:
        print(f"Processed samples: {fold + 1}/{total_folds}...")

y_true_all = np.array(y_true_all).squeeze(1)
y_pred_all = np.array(y_pred_all).squeeze(1)

final_pikan_r2 = r2_score(y_true_all, y_pred_all, multioutput='uniform_average')

if final_pikan_r2 > 0.60:
    print("Training final model on 100% of data...")
    final_model = AdvancedPINN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.005, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)
    
    epochs = 2000
    for epoch in range(1, epochs + 1):
        final_model.train()
        optimizer.zero_grad()
        predictions = final_model(X)
        loss_data = criterion(predictions, Y)
        
        X_phys.requires_grad_(True)
        pred_phys = final_model(X_phys)
        grad_E = torch.autograd.grad(outputs=pred_phys[:, 0], inputs=X_phys, grad_outputs=torch.ones_like(pred_phys[:, 0]).to(device), create_graph=True)[0]
        grad_UTS = torch.autograd.grad(outputs=pred_phys[:, 1], inputs=X_phys, grad_outputs=torch.ones_like(pred_phys[:, 1]).to(device), create_graph=True)[0]
        loss_physics = (
            torch.mean(torch.relu(-grad_E[:, 0])) + torch.mean(torch.relu(-grad_E[:, 1])) +
            torch.mean(torch.relu(-grad_UTS[:, 0])) + torch.mean(torch.relu(-grad_UTS[:, 1]))
        )
        lambda_p = min(0.5, 0.01 * (epoch / 100))
        total_loss = loss_data + lambda_p * loss_physics
        total_loss.backward()
        optimizer.step()
        scheduler.step(loss_data)
        
    if not os.path.exists('./best_model'):
        os.makedirs('./best_model')
    torch.save(final_model.state_dict(), './best_model/best_pinn_model.pth')

if hasattr(data_prop, 'scaler_y'):
    y_true_real = data_prop.scaler_y.inverse_transform(y_true_all)
    y_pred_real = data_prop.scaler_y.inverse_transform(y_pred_all)
else:
    if hasattr(data_prop, 'get_raw_data'):
        _, Y_raw = data_prop.get_raw_data()
        y_min = Y_raw.min(axis=0)
        y_max = Y_raw.max(axis=0)
    else:
        y_min = np.array([2.0, 50.0, 1.0, 40.0])   
        y_max = np.array([15.0, 180.0, 30.0, 140.0]) 
    y_true_real = 0.5 * (y_true_all + 1) * (y_max - y_min) + y_min
    y_pred_real = 0.5 * (y_pred_all + 1) * (y_max - y_min) + y_min

units = ['GPa', 'MPa', '%', 'MPa']
property_names = ['Young Modulus', 'UTS', 'Elongation', 'Yield Strength']

print("\n" + "="*95)
print("                               PINN LOOCV VALIDATION METRICS RESULTS")
print("===============================================================================================")
print(f"Overall Multioutput PINN LOOCV R2 Score: {final_pikan_r2:.4f}")
print("-"*95)

for i, name in enumerate(property_names):
    r2 = r2_score(y_true_real[:, i], y_pred_real[:, i])
    mae = mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(y_true_real[:, i], y_pred_real[:, i]))
    mape = np.mean(np.abs((y_true_real[:, i] - y_pred_real[:, i]) / y_true_real[:, i])) * 100
    print(f"Property: {name:<15} | R2: {r2:>7.4f} | MAE: {mae:>7.2f} {units[i]:<3} | RMSE: {rmse:>7.2f} {units[i]:<3} | MAPE: {mape:>6.2f}%")
print("=" * 95)

plt.figure(figsize=(10, 5))
for i, name in enumerate(property_names):
    plt.scatter(y_true_real[:, i], y_pred_real[:, i], alpha=0.7, label=f"{name} ({units[i]})")
min_val = min(y_true_real.min(), y_pred_real.min())
max_val = max(y_true_real.max(), y_pred_real.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')
plt.xlabel('Experimental True (Physical Units)')
plt.ylabel('PINN Predicted (Physical Units)')
plt.title('PINN LOOCV True vs Predicted Values (Real Scales)')
plt.legend()
plt.grid(True)
plt.show()
