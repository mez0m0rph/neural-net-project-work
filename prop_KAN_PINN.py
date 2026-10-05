import torch
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import prop_data_2_lg as data_prop
import os
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X, Y = data_prop.get_pytorch_data(device)
X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()

X_phys = data_prop.get_physics_point(1000).to(device)

loo = LeaveOneOut()

y_true_all = []
y_pred_all = []

lambda_p = 0.05
hidden_neurons = 34

for fold, (train_idx, val_idx) in enumerate(loo.split(X_np)):
    X_train, X_val = X_np[train_idx], X_np[val_idx]
    Y_train, Y_val = Y_np[train_idx], Y_np[val_idx]
    
    dataset = {
        'train_input': torch.tensor(X_train, dtype=torch.float32).to(device),
        'train_label': torch.tensor(Y_train, dtype=torch.float32).to(device),
        'test_input': torch.tensor(X_val, dtype=torch.float32).to(device),
        'test_label': torch.tensor(Y_val, dtype=torch.float32).to(device)
    }
    
    model = KAN(width=[2, hidden_neurons, 4], grid=4, k=3, device=str(device))
    
    def pikan_loss(pred_data, true_data):
        loss_data = torch.nn.functional.mse_loss(pred_data, true_data)
        
        X_phys.requires_grad_(True)
        pred_phys = model(X_phys)
        E_pred = pred_phys[:, 0]
        UTS_pred = pred_phys[:, 1]
        
        grad_E = torch.autograd.grad(
            outputs=E_pred,
            inputs=X_phys,
            grad_outputs=torch.ones_like(E_pred),
            create_graph=True
        )[0]
        
        grad_UTS = torch.autograd.grad(
            outputs=UTS_pred,
            inputs=X_phys,
            grad_outputs=torch.ones_like(UTS_pred),
            create_graph=True
        )[0]
        
        dE_dWcf = grad_E[:, 0]
        dE_dlcf = grad_E[:, 1]
        dUTS_dWcf = grad_UTS[:, 0]
        dUTS_dlcf = grad_UTS[:, 1]
        
        loss_physics = (
            torch.mean(torch.nn.functional.relu(-dE_dWcf)) +
            torch.mean(torch.nn.functional.relu(-dE_dlcf)) +
            torch.mean(torch.nn.functional.relu(-dUTS_dWcf)) +
            torch.mean(torch.nn.functional.relu(-dUTS_dlcf))
        )
        
        return loss_data + lambda_p * loss_physics

    sys.stdout = open(os.devnull, 'w')
    model.fit(dataset, opt="Adam", steps=50, lr=0.05, loss_fn=pikan_loss)
    sys.stdout = sys.__stdout__
    
    with torch.no_grad():
        val_pred = model(dataset['test_input']).cpu().numpy()
        
    y_true_all.append(Y_val[0])
    y_pred_all.append(val_pred[0])

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

final_pikan_r2 = r2_score(y_true_all, y_pred_all, multioutput='uniform_average')

if final_pikan_r2 > 0.60:
    full_dataset = {
        'train_input': X, 'train_label': Y,
        'test_input': X, 'test_label': Y
    }
    final_model = KAN(width=[2, hidden_neurons, 4], grid=4, k=3, device=str(device))
    
    def final_pikan_loss(pred_data, true_data):
        loss_data = torch.nn.functional.mse_loss(pred_data, true_data)
        X_phys.requires_grad_(True)
        pred_phys = final_model(X_phys)
        E_pred = pred_phys[:, 0]
        UTS_pred = pred_phys[:, 1]
        
        grad_E = torch.autograd.grad(outputs=E_pred, inputs=X_phys, grad_outputs=torch.ones_like(E_pred), create_graph=True)[0]
        grad_UTS = torch.autograd.grad(outputs=UTS_pred, inputs=X_phys, grad_outputs=torch.ones_like(UTS_pred), create_graph=True)[0]
        
        loss_physics = (
            torch.mean(torch.nn.functional.relu(-grad_E[:, 0])) +
            torch.mean(torch.nn.functional.relu(-grad_E[:, 1])) +
            torch.mean(torch.nn.functional.relu(-grad_UTS[:, 0])) +
            torch.mean(torch.nn.functional.relu(-grad_UTS[:, 1]))
        )
        return loss_data + lambda_p * loss_physics

    sys.stdout = open(os.devnull, 'w')
    final_model.fit(full_dataset, opt="Adam", steps=50, lr=0.05, loss_fn=final_pikan_loss)
    sys.stdout = sys.__stdout__
    
    if not os.path.exists('./best_model'):
        os.makedirs('./best_model')
    torch.save(final_model.state_dict(), './best_model/best_pikan_model.pth')

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
print("                               PIKAN LOOCV VALIDATION METRICS RESULTS")
print("===============================================================================================")
print(f"Overall Multioutput PIKAN LOOCV R2 Score: {final_pikan_r2:.4f}")
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
plt.ylabel('PIKAN Predicted (Physical Units)')
plt.title('PIKAN LOOCV True vs Predicted Values (Real Scales)')
plt.legend()
plt.grid(True)
plt.show()
