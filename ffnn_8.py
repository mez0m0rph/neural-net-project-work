
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import prop_data_2_lg_real as data_prop


SEED = 42
EPOCHS = 1500
LEARNING_RATE = 0.01
SYNTHETIC_LENGTHS = [0.2, 0.5, 1.0, 3.0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

property_names = [
    "Young Modulus",
    "UTS",
    "Elongation",
    "Yield Strength"
]

units = ["GPa", "MPa", "%", "MPa"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class FFNN8(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.network(x)


def log_targets(y):
    result = np.asarray(y, dtype=np.float64).copy()
    result[:, 2] = np.log10(result[:, 2])
    return result


def augment_training_data(X, Y):
    X_aug = list(np.asarray(X, dtype=np.float64))
    Y_aug = list(np.asarray(Y, dtype=np.float64))

    pure_peek = np.all(
        np.isclose(X, np.array([0.0, 0.0])),
        axis=1
    )

    for y in np.asarray(Y)[pure_peek]:
        for length in SYNTHETIC_LENGTHS:
            X_aug.append([0.0, length])
            Y_aug.append(y.copy())

    return np.asarray(X_aug), np.asarray(Y_aug)


def restore_targets(y_scaled, scaler_y):
    result = scaler_y.inverse_transform(y_scaled)
    result[:, 2] = np.power(10.0, result[:, 2])
    return result


def train_model(X_train, Y_train, seed):
    set_seed(seed)

    model = FFNN8().to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32,
        device=device
    )

    Y_tensor = torch.tensor(
        Y_train,
        dtype=torch.float32,
        device=device
    )

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        predictions = model(X_tensor)
        loss = criterion(predictions, Y_tensor)

        loss.backward()
        optimizer.step()

    return model


set_seed(SEED)

X_base = np.asarray(data_prop.X_base, dtype=np.float64)
Y_base = np.asarray(data_prop.Y_base, dtype=np.float64)

_, groups = np.unique(
    X_base,
    axis=0,
    return_inverse=True
)

cv = LeaveOneGroupOut()

y_pred_cv = np.full_like(Y_base, np.nan, dtype=np.float64)

print(f"Device: {device}")
print(f"Experimental samples: {len(X_base)}")
print(f"Unique compositions: {len(np.unique(groups))}")
print("Starting leave-one-composition-out validation...")

for fold, (train_idx, val_idx) in enumerate(
    cv.split(X_base, Y_base, groups),
    start=1
):
    X_train_raw, Y_train_raw = augment_training_data(
        X_base[train_idx],
        Y_base[train_idx]
    )

    scaler_x_fold = MinMaxScaler()
    scaler_y_fold = MinMaxScaler()

    X_train = scaler_x_fold.fit_transform(X_train_raw)
    Y_train = scaler_y_fold.fit_transform(log_targets(Y_train_raw))

    X_val = scaler_x_fold.transform(X_base[val_idx])

    model = train_model(
        X_train,
        Y_train,
        SEED + fold
    )

    model.eval()

    X_val_tensor = torch.tensor(
        X_val,
        dtype=torch.float32,
        device=device
    )

    with torch.no_grad():
        predictions_scaled = model(X_val_tensor).cpu().numpy()

    y_pred_cv[val_idx] = restore_targets(
        predictions_scaled,
        scaler_y_fold
    )

    print(f"Completed group {fold}/{len(np.unique(groups))}")

y_true_real = Y_base
y_pred_real = y_pred_cv

overall_r2 = r2_score(
    y_true_real,
    y_pred_real,
    multioutput="uniform_average"
)

print("\n" + "=" * 95)
print("FFNN 8: LEAVE-ONE-COMPOSITION-OUT VALIDATION")
print("=" * 95)
print(f"Overall multioutput R2: {overall_r2:.4f}")
print("-" * 95)

for i, name in enumerate(property_names):
    r2 = r2_score(y_true_real[:, i], y_pred_real[:, i])
    mae = mean_absolute_error(y_true_real[:, i], y_pred_real[:, i])
    rmse = np.sqrt(
        mean_squared_error(y_true_real[:, i], y_pred_real[:, i])
    )
    mape = np.mean(
        np.abs(
            (y_true_real[:, i] - y_pred_real[:, i])
            / y_true_real[:, i]
        )
    ) * 100

    print(
        f"{name:<17} | R2: {r2:>8.4f} | "
        f"MAE: {mae:>8.3f} {units[i]:<3} | "
        f"RMSE: {rmse:>8.3f} {units[i]:<3} | "
        f"MAPE: {mape:>7.2f}%"
    )

print("=" * 95)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.scatter(
        y_true_real[:, i],
        y_pred_real[:, i],
        alpha=0.75,
        edgecolors="black",
        linewidths=0.4
    )

    min_val = min(
        y_true_real[:, i].min(),
        y_pred_real[:, i].min()
    )
    max_val = max(
        y_true_real[:, i].max(),
        y_pred_real[:, i].max()
    )

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=1.5
    )

    ax.set_title(property_names[i])
    ax.set_xlabel(f"Experimental value ({units[i]})")
    ax.set_ylabel(f"Predicted value ({units[i]})")
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


X_full, Y_full = data_prop.get_pytorch_data(device)

set_seed(SEED)

final_model = FFNN8().to(device)
optimizer = optim.Adam(
    final_model.parameters(),
    lr=LEARNING_RATE
)
criterion = nn.MSELoss()

for epoch in range(EPOCHS):
    final_model.train()
    optimizer.zero_grad()

    predictions = final_model(X_full)
    loss = criterion(predictions, Y_full)

    loss.backward()
    optimizer.step()

os.makedirs("./best_model", exist_ok=True)

torch.save(
    final_model.state_dict(),
    "./best_model/best_ffnn_8_model.pth"
)

print("\nFinal FFNN8 model saved to ./best_model/best_ffnn_8_model.pth")