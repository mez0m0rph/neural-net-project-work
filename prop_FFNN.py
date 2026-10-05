import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
import prop_data_2 as data_prop

import random
import os

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
print(f"Используется устройство для расчетов: {device}/n")

X,Y = data_prop.get_pytorch_data(device)

X_train, Y_train = X[:12], Y[:12]
X_val, Y_val = X[12:], Y[12:]

print(f"Train size: {X_train.shape}")
print(f"Val size: {X_val.shape}\n")

class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 4)
        )
        
    def forward(self, x):
        return self.network(x)

model = FeedForwardNN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

history_train_loss = []
history_val_loss = []

epochs = 1500
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train)
    loss = criterion(predictions, Y_train)

    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, Y_val)

    history_train_loss.append(loss.item())
    history_val_loss.append(val_loss.item())

    if epoch % 300 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f}")

model.eval()
with torch.no_grad():
    final_pred = model(X_val)
    y_true = Y_val.cpu().numpy()
    y_pred = final_pred.cpu().numpy()
    final_r2 = r2_score(y_true, y_pred)
    print(f"\nFinal Val R2 Score: {final_r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(history_train_loss, label='Train Loss')
plt.plot(history_val_loss, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('FFNN Training History')
plt.legend()
plt.grid(True)
plt.show()