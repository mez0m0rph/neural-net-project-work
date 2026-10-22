
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import prop_data_2_lg_real as data_prop
from model_architectures import (
    FFNNTwoHidden25,
    FFNNSingleHidden16,
    FFNNSingleHidden8
)


SEED = 42
EPOCHS = 1500
LEARNING_RATE = 0.01
MODEL_DIR = "./best_model"

units = ["GPa", "MPa", "%", "MPa"]
property_names = [
    "Young Modulus",
    "UTS",
    "Elongation",
    "Yield Strength"
]


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def to_physical_units(y_scaled):
    y_physical = data_prop.scaler_Y.inverse_transform(y_scaled)
    y_physical[:, 2] = np.power(10.0, y_physical[:, 2])
    return y_physical


def fit_model(model_class, X_train, Y_train, seed):
    set_seed(seed)

    model = model_class().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    X_train_t = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device
    )

    Y_train_t = torch.tensor(
        Y_train,
        dtype=torch.float32,
        device=device
    )

    for _ in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_train_t)
        loss = criterion(predictions, Y_train_t)

        loss.backward()
        optimizer.step()

    return model


def plot_validation(model_name, y_true, y_pred):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.scatter(
            y_true[:, i],
            y_pred[:, i],
            alpha=0.75,
            edgecolors="black",
            linewidths=0.35
        )

        lower = min(
            y_true[:, i].min(),
            y_pred[:, i].min()
        )
        upper = max(
            y_true[:, i].max(),
            y_pred[:, i].max()
        )

        ax.plot(
            [lower, upper],
            [lower, upper],
            "r--",
            linewidth=1.4
        )

        ax.set_title(property_names[i])
        ax.set_xlabel(f"Experimental value ({units[i]})")
        ax.set_ylabel(f"Predicted value ({units[i]})")
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{model_name}: LOOCV predictions",
        fontsize=14
    )

    fig.tight_layout()
    plt.show()


def evaluate_model(model_class, model_name, X_np, Y_np):
    loo = LeaveOneOut()

    y_true_norm = np.empty_like(Y_np)
    y_pred_norm = np.empty_like(Y_np)

    print(f"\nStarting LOOCV for {model_name}...")

    for fold, (train_idx, val_idx) in enumerate(
        loo.split(X_np),
        start=1
    ):
        model = fit_model(
            model_class,
            X_np[train_idx],
            Y_np[train_idx],
            SEED + fold
        )

        X_val_t = torch.tensor(
            X_np[val_idx],
            dtype=torch.float32,
            device=device
        )

        model.eval()

        with torch.no_grad():
            y_pred_norm[val_idx] = (
                model(X_val_t).cpu().numpy()
            )

        y_true_norm[val_idx] = Y_np[val_idx]

        if fold % 10 == 0 or fold == len(X_np):
            print(f"Processed {fold}/{len(X_np)} folds")

    overall_r2 = r2_score(
        y_true_norm,
        y_pred_norm,
        multioutput="uniform_average"
    )

    y_true_real = to_physical_units(y_true_norm)
    y_pred_real = to_physical_units(y_pred_norm)

    print("\n" + "=" * 100)
    print(f"{model_name} LOOCV RESULTS")
    print("=" * 100)
    print(f"Overall multioutput R2: {overall_r2:.4f}")
    print("-" * 100)

    for i, name in enumerate(property_names):
        r2 = r2_score(
            y_true_real[:, i],
            y_pred_real[:, i]
        )

        mae = mean_absolute_error(
            y_true_real[:, i],
            y_pred_real[:, i]
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_true_real[:, i],
                y_pred_real[:, i]
            )
        )

        denominator = np.maximum(
            np.abs(y_true_real[:, i]),
            1e-12
        )

        mape = np.mean(
            np.abs(
                y_true_real[:, i] - y_pred_real[:, i]
            ) / denominator
        ) * 100

        print(
            f"{name:<17} | R2: {r2:>8.4f} | "
            f"MAE: {mae:>8.3f} {units[i]:<3} | "
            f"RMSE: {rmse:>8.3f} {units[i]:<3} | "
            f"MAPE: {mape:>7.2f}%"
        )

    print("=" * 100)

    plot_validation(
        model_name,
        y_true_real,
        y_pred_real
    )

    return overall_r2


def train_and_save_full_model(
    model_class,
    model_name,
    filename,
    X_np,
    Y_np
):
    final_model = fit_model(
        model_class,
        X_np,
        Y_np,
        SEED
    )

    final_model.eval()

    os.makedirs(MODEL_DIR, exist_ok=True)

    path = os.path.join(MODEL_DIR, filename)

    torch.save(
        final_model.state_dict(),
        path
    )

    print(f"Saved {model_name} to {path}")

    return final_model


set_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Device: {device}")

X, Y = data_prop.get_pytorch_data(device)

X_np = X.detach().cpu().numpy()
Y_np = Y.detach().cpu().numpy()

experiments = [
    (
        "FFNN: 2 hidden layers x 25 neurons",
        FFNNTwoHidden25,
        "ffnn_2x25_model.pth"
    ),
    (
        "FFNN: 1 hidden layer x 16 neurons",
        FFNNSingleHidden16,
        "ffnn_1x16_model.pth"
    ),
    (
        "FFNN: 1 hidden layer x 8 neurons",
        FFNNSingleHidden8,
        "ffnn_1x8_model.pth"
    )
]

for model_name, model_class, filename in experiments:
    evaluate_model(
        model_class,
        model_name,
        X_np,
        Y_np
    )

    train_and_save_full_model(
        model_class,
        model_name,
        filename,
        X_np,
        Y_np
    )

base_model_path = os.path.join(
    MODEL_DIR,
    "ffnn_2x25_model.pth"
)

legacy_model_path = os.path.join(
    MODEL_DIR,
    "best_ffnn_model.pth"
)

if os.path.isfile(base_model_path):
    state_dict = torch.load(
        base_model_path,
        map_location=device
    )

    torch.save(
        state_dict,
        legacy_model_path
    )

    print(
        f"Saved compatibility copy to {legacy_model_path}"
    )