import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from kan import KAN
import prop_data_2_lg as data_prop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

kan_model = KAN(width=[2, 34, 4], grid=5, k=3, device=str(device))
pikan_model = KAN(width=[2, 34, 4], grid=4, k=3, device=str(device))

kan_model.load_state_dict(torch.load('./best_model/best_kan_model.pth', map_location=device))
pikan_model.load_state_dict(torch.load('./best_model/best_pikan_model.pth', map_location=device))

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

ffnn_model = FeedForwardNN().to(device)
pinn_model = AdvancedPINN().to(device)

ffnn_model.load_state_dict(torch.load('./best_model/best_ffnn_model.pth', map_location=device))
pinn_model.load_state_dict(torch.load('./best_model/best_pinn_model.pth', map_location=device))

kan_model.eval()
pikan_model.eval()
ffnn_model.eval()
pinn_model.eval()

property_names = ['Young Modulus', 'UTS', 'Elongation', 'Yield Strength']
units = ['GPa', 'MPa', '%', 'MPa']

X_exp = data_prop.X_raw.copy()
Y_exp = data_prop.Y_raw.copy()
Y_log = Y_exp.copy()
Y_log[:, 2] = np.log10(Y_log[:, 2])
y_mins = Y_log.min(axis=0)
y_maxs = Y_log.max(axis=0)

cf_range = np.linspace(0.0, 60.0, 100)
length_range = np.linspace(0.1, 3.5, 100)  

CF_mesh, LV_mesh = np.meshgrid(cf_range, length_range)

X_grid_raw = np.vstack([CF_mesh.ravel(), LV_mesh.ravel()]).T
X_grid_scaled = data_prop.scaler_X.transform(X_grid_raw)
X_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    y_kan_sc = kan_model(X_tensor).cpu().numpy()
    y_pikan_sc = pikan_model(X_tensor).cpu().numpy()
    y_ffnn_sc = ffnn_model(X_tensor).cpu().numpy()
    y_pinn_sc = pinn_model(X_tensor).cpu().numpy()

preds = {'Base KAN': y_kan_sc, 'PIKAN': y_pikan_sc, 'Base FFNN': y_ffnn_sc, 'PINN': y_pinn_sc}
denorm_preds = {}

for name, y_sc in preds.items():
    y_denorm = np.zeros_like(y_sc)
    for col in range(4):
        y_denorm[:, col] = y_sc[:, col] * (y_maxs[col] - y_mins[col]) + y_mins[col]
    y_denorm[:, 2] = 10 ** y_denorm[:, 2]
    denorm_preds[name] = y_denorm

models_list = ['Base KAN', 'PIKAN', 'Base FFNN', 'PINN']

for prop_idx in range(4):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    all_vals = np.concatenate([denorm_preds[m][:, prop_idx] for m in models_list])
    levels = np.linspace(all_vals.min(), all_vals.max(), 15)
    
    for ax_idx, model_name in enumerate(models_list):
        ax = axes[ax_idx]
        
        Z = denorm_preds[model_name][:, prop_idx].reshape(CF_mesh.shape)
        
        contour_f = ax.contourf(CF_mesh, LV_mesh, Z, levels=levels, cmap='viridis', alpha=0.85)
        contour_l = ax.contour(CF_mesh, LV_mesh, Z, levels=levels, colors='black', linewidths=0.5)
        ax.clabel(contour_l, inline=True, fontsize=8, fmt='%.1f')
        
        ax.scatter(X_exp[:, 0], X_exp[:, 1], color='red', edgecolors='white', s=25, label='Exp Points', zorder=5)
        
        ax.axvspan(50.0, 60.0, color='red', alpha=0.08, hatch='//')
        ax.axvline(x=50.0, color='red', linestyle=':', alpha=0.5)
        
        ax.set_title(model_name, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        if ax_idx >= 2:
            ax.set_xlabel('CF Content (%)')
        if ax_idx % 2 == 0:
            ax.set_ylabel('Fiber Length (mm)')
            
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contour_f, cax=cbar_ax)
    cbar.set_label(f'{property_names[prop_idx]} ({units[prop_idx]})', fontsize=12, fontweight='bold')
    
    plt.suptitle(f'Contour Plots for {property_names[prop_idx]} Across Models\n(Hatched area = Extrapolation Zone)', 
                 fontsize=16, fontweight='bold', y=0.96)
    
    plt.subplots_adjust(right=0.90, hspace=0.2, wspace=0.1)
    plt.show()