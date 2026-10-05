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

experimental_lengths = [0.2, 0.5, 1.0, 3.0]
cf_extrapolated = np.linspace(0.0, 60.0, 200)

property_names = ['Young Modulus', 'UTS', 'Elongation', 'Yield Strength']
units = ['GPa', 'MPa', '%', 'MPa']

X_exp = data_prop.X_raw.copy()
Y_exp = data_prop.Y_raw.copy()

Y_log = Y_exp.copy()
Y_log[:, 2] = np.log10(Y_log[:, 2])

y_mins = Y_log.min(axis=0)
y_maxs = Y_log.max(axis=0)

for length in experimental_lengths:
    X_slice_raw = np.zeros((200, 2))
    X_slice_raw[:, 0] = cf_extrapolated
    X_slice_raw[:, 1] = length
    
    X_slice_scaled = data_prop.scaler_X.transform(X_slice_raw)
    X_tensor = torch.tensor(X_slice_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_kan_sc = kan_model(X_tensor).cpu().numpy()
        y_pikan_sc = pikan_model(X_tensor).cpu().numpy()
        y_ffnn_sc = ffnn_model(X_tensor).cpu().numpy()
        y_pinn_sc = pinn_model(X_tensor).cpu().numpy()
        
    y_kan = np.zeros_like(y_kan_sc)
    y_pikan = np.zeros_like(y_pikan_sc)
    y_ffnn = np.zeros_like(y_ffnn_sc)
    y_pinn = np.zeros_like(y_pinn_sc)
    
    for col in range(4):
        y_kan[:, col] = y_kan_sc[:, col] * (y_maxs[col] - y_mins[col]) + y_mins[col]
        y_pikan[:, col] = y_pikan_sc[:, col] * (y_maxs[col] - y_mins[col]) + y_mins[col]
        y_ffnn[:, col] = y_ffnn_sc[:, col] * (y_maxs[col] - y_mins[col]) + y_mins[col]
        y_pinn[:, col] = y_pinn_sc[:, col] * (y_maxs[col] - y_mins[col]) + y_mins[col]
        
    y_kan[:, 2] = 10 ** y_kan[:, 2]
    y_pikan[:, 2] = 10 ** y_pikan[:, 2]
    y_ffnn[:, 2] = 10 ** y_ffnn[:, 2]
    y_pinn[:, 2] = 10 ** y_pinn[:, 2]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    indices_to_plot = np.where(X_exp[:, 1] == length)
        
    for i in range(4):
        ax = axes[i]
        
        ax.plot(cf_extrapolated, y_kan[:, i], label='Base KAN', color='blue', linewidth=2)
        ax.plot(cf_extrapolated, y_pikan[:, i], label='PIKAN (Physics KAN)', color='cyan', linewidth=2.5)
        ax.plot(cf_extrapolated, y_ffnn[:, i], label='Base FFNN', color='green', linewidth=1.5, linestyle='--')
        ax.plot(cf_extrapolated, y_pinn[:, i], label='PINN (Physics MLP)', color='orange', linewidth=2, linestyle='-.')
        
        ax.scatter(X_exp[indices_to_plot, 0], Y_exp[indices_to_plot, i], color='red', alpha=0.7, s=40, zorder=5, label='Experimental Points')
        
        ax.axvspan(50.0, 60.0, color='gray', alpha=0.15, label='Extrapolation Zone')
        ax.axvline(x=50.0, color='red', linestyle=':', alpha=0.7)
        
        if i == 2:
            ax.set_ylim(-1, 45.0)
            ax.set_yticks(np.arange(0, 46, 5))
            
        ax.set_title(f'{property_names[i]} ({units[i]})', fontsize=12, fontweight='bold')
        ax.set_xlabel('CF Content (%)')
        ax.set_ylabel(f'{property_names[i]}')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
    plt.suptitle(f'Mechanical Properties Trends & Extrapolation for Fixed Fiber Length = {length} mm', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
