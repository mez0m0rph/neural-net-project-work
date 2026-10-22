
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from kan import KAN

import prop_data_2_lg_real as data_real
import prop_data_2_lg as data_lg


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = "./best_model"
GRID_RESOLUTION = 150

property_names = [
    "Young Modulus",
    "UTS",
    "Elongation",
    "Yield Strength"
]

units = ["GPa", "MPa", "%", "MPa"]

model_names = ["Base KAN", "PIKAN", "Base FFNN", "PINN"]

print(f"Device: {device}")


class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 16),
            nn.Tanh(),
            nn.Linear(16, 4)
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


def get_scaler(module, *names):
    for name in names:
        scaler = getattr(module, name, None)
        if scaler is not None:
            return scaler

    raise AttributeError(
        f"Scaler not found in {module.__name__}: {names}"
    )


def inverse_predictions(predictions, module):
    scaler_y = get_scaler(module, "scaler_Y", "scaler_y")
    predictions_real = scaler_y.inverse_transform(predictions)

    with np.errstate(over="ignore", invalid="ignore"):
        predictions_real[:, 2] = np.power(
            10.0,
            predictions_real[:, 2]
        )

    predictions_real[~np.isfinite(predictions_real)] = np.nan

    return predictions_real


kan_model = KAN(
    width=[2, 34, 4],
    grid=5,
    k=3,
    device=str(device)
)

pikan_model = KAN(
    width=[2, 34, 4],
    grid=4,
    k=3,
    device=str(device)
)

ffnn_model = FeedForwardNN().to(device)
pinn_model = AdvancedPINN().to(device)

models = {
    "Base KAN": (kan_model, data_real),
    "PIKAN": (pikan_model, data_lg),
    "Base FFNN": (ffnn_model, data_real),
    "PINN": (pinn_model, data_lg)
}

model_files = {
    "Base KAN": "best_kan_model.pth",
    "PIKAN": "best_pikan_model.pth",
    "Base FFNN": "best_ffnn_model.pth",
    "PINN": "best_pinn_model.pth"
}

for name, (model, _) in models.items():
    path = os.path.join(MODEL_DIR, model_files[name])

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Model weights not found: {path}")

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Loaded: {name}")


X_exp = np.unique(data_real.X_base, axis=0)

cf_range = np.linspace(0.0, 60.0, GRID_RESOLUTION)
length_range = np.linspace(0.0, 3.5, GRID_RESOLUTION)

CF_mesh, LV_mesh = np.meshgrid(cf_range, length_range)

X_grid_raw = np.column_stack([
    CF_mesh.ravel(),
    LV_mesh.ravel()
])

predictions = {}

for name, (model, module) in models.items():
    scaler_x = get_scaler(module, "scaler_X", "scaler_x")
    X_grid_scaled = scaler_x.transform(X_grid_raw)

    X_tensor = torch.tensor(
        X_grid_scaled,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        pred_scaled = model(X_tensor).cpu().numpy()

    predictions[name] = inverse_predictions(
        pred_scaled,
        module
    )

    print(f"Predictions generated: {name}")


for prop_idx, prop_name in enumerate(property_names):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15, 10),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes = axes.flatten()

    all_values = np.concatenate([
        predictions[name][:, prop_idx]
        for name in model_names
    ])

    all_values = all_values[np.isfinite(all_values)]

    if all_values.size == 0:
        plt.close(fig)
        raise ValueError(
            f"No finite predictions available for {prop_name}"
        )

    vmin, vmax = np.percentile(all_values, [2, 98])

    if np.isclose(vmin, vmax):
        padding = max(abs(vmin) * 0.01, 1e-6)
        vmin -= padding
        vmax += padding

    levels = np.linspace(vmin, vmax, 16)
    contour = None

    for ax_idx, (ax, model_name) in enumerate(
        zip(axes, model_names)
    ):
        Z = predictions[model_name][:, prop_idx].reshape(
            CF_mesh.shape
        )

        contour = ax.contourf(
            CF_mesh,
            LV_mesh,
            Z,
            levels=levels,
            cmap="viridis",
            extend="both"
        )

        lines = ax.contour(
            CF_mesh,
            LV_mesh,
            Z,
            levels=levels[1:-1:2],
            colors="black",
            linewidths=0.45,
            alpha=0.6
        )

        ax.clabel(
            lines,
            inline=True,
            fontsize=7,
            fmt="%.2g"
        )

        ax.scatter(
            X_exp[:, 0],
            X_exp[:, 1],
            color="red",
            edgecolors="white",
            linewidths=0.7,
            s=28,
            zorder=5
        )

        ax.axvspan(
            50.0,
            60.0,
            facecolor="red",
            alpha=0.07,
            hatch="//",
            edgecolor="red",
            linewidth=0.0
        )

        ax.axhspan(
            3.0,
            3.5,
            facecolor="orange",
            alpha=0.09,
            hatch="\\\\",
            edgecolor="orange",
            linewidth=0.0
        )

        ax.set_title(
            model_name,
            fontsize=12,
            fontweight="bold"
        )

        ax.set_xlim(0.0, 60.0)
        ax.set_ylim(0.0, 3.5)

        ax.xaxis.set_major_locator(MaxNLocator(6))
        ax.yaxis.set_major_locator(MaxNLocator(6))

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.4,
            alpha=0.25
        )

        if ax_idx >= 2:
            ax.set_xlabel("CF Content (%)", fontsize=10)

        if ax_idx % 2 == 0:
            ax.set_ylabel("Fiber Length (mm)", fontsize=10)

    cbar = fig.colorbar(
        contour,
        ax=axes.tolist(),
        location="right",
        shrink=0.90,
        pad=0.025
    )

    cbar.set_label(
        f"{prop_name} ({units[prop_idx]})",
        fontsize=11
    )

    legend_handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="w",
            markerfacecolor="red",
            markeredgecolor="white",
            markersize=7,
            label="Experimental compositions"
        ),
        Patch(
            facecolor="red",
            alpha=0.12,
            hatch="//",
            label="CF > 50%"
        ),
        Patch(
            facecolor="orange",
            alpha=0.15,
            hatch="\\\\",
            label="Fiber length > 3 mm"
        )
    ]

    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=3,
        frameon=False,
        fontsize=9
    )

    fig.suptitle(
        f"{prop_name}: Comparison of Four Neural Network Models",
        fontsize=15,
        fontweight="bold"
    )

    plt.show()
