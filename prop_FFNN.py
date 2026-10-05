import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import prop_data_2_lg as data_prop
import os
import sys
import random

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

X, Y = data_prop.get_pytorch_data(device)
X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()

loo = LeaveOneOut()

y_true_all = []
y_pred_all = []

class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 25),
            nn.Tanh(),
            nn.Linear(25, 25),
            nn.Tanh(),
            nn.Linear(25, 4)
        )
        
    def forward(self, x):
        return self.network(x)

for fold, (train_idx, val_idx) in enumerate(loo.split(X_np)):
    X_train, X_val = X_np[train_idx], X_np[val_idx]
    Y_train, Y_val = Y_np[train_idx], Y_np[val_idx]
    
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    
    model = FeedForwardNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epochs = 1500
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        predictions = model(X_train_t)
        loss = criterion(predictions, Y_train_t)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_t).cpu().numpy()
        
    y_true_all.append(Y_val[0])
    y_pred_all.append(val_pred[0])

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

final_loocv_r2 = r2_score(y_true_all, y_pred_all, multioutput='uniform_average')

if final_loocv_r2 > 0.60:
    final_model = FeedForwardNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(final_model.parameters(), lr=0.01)
    
    epochs = 1500
    for epoch in range(1, epochs + 1):
        final_model.train()
        optimizer.zero_grad()
        predictions = final_model(X)
        loss = criterion(predictions, Y)
        loss.backward()
        optimizer.step()
        
    if not os.path.exists('./best_model'):
        os.makedirs('./best_model')
    torch.save(final_model.state_dict(), './best_model/best_ffnn_model.pth')

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
print("                               FFNN LOOCV VALIDATION METRICS RESULTS")
print("===============================================================================================")
print(f"Overall Multioutput FFNN LOOCV R2 Score: {final_loocv_r2:.4f}")
print("-"*95)

for i, name in enumerate(property_names):
    r2 = r2_score(y_true_real[:, i], y_pred_real[:, i])
    mae = mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])
    rmse = np.sqrt(mean_squared_error(y_true_real[:, i], y_pred_real[:, i]))
    mape = np.mean(np.abs((y_true_real[:, i] - y_pred_real[:, i]) / y_true_real[:, i])) * 100
    print(f"Property: {name:<15} | R2: {r2:>7.4f} | MAE: {mae:>7.2f} {units[i]:<3} | RMSE: {rmse:>7.2f} {units[i]:<3} | MAPE: {mape:>6.2f}%")
print("="*95)

plt.figure(figsize=(10, 5))
for i, name in enumerate(property_names):
    plt.scatter(y_true_real[:, i], y_pred_real[:, i], alpha=0.7, label=f"{name} ({units[i]})")

min_val = min(y_true_real.min(), y_pred_real.min())
max_val = max(y_true_real.max(), y_pred_real.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal Prediction')

plt.xlabel('Experimental True (Physical Units)')
plt.ylabel('FFNN Predicted (Physical Units)')
plt.title('FFNN LOOCV True vs Predicted Values (Real Scales)')
plt.legend()
plt.grid(True)
plt.show()