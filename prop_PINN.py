import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import prop_data_2 as data_prop
import numpy as np

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
print(f"Device: {device}")

X, Y = data_prop.get_pytorch_data(device)

X_train, Y_train = X[:12], Y[:12]
X_val, Y_val = X[12:], Y[12:]

class HardConstrainedPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 4)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        identity = x
        
        h = self.fc1(x)
        h = self.tanh(h)
        h = self.dropout(h)
        
        h = self.fc2(h)
        h = self.tanh(h)
        h = self.dropout(h)
        
        raw_out = self.output(h)
        
        E_constrained = torch.abs(raw_out[:, 0:1]) * identity[:, 0:1] + raw_out[:, 0:1]
        UTS_constrained = torch.abs(raw_out[:, 1:2]) * identity[:, 0:1] + raw_out[:, 1:2]
        Elong_constrained = raw_out[:, 2:3]
        Yield_constrained = torch.abs(raw_out[:, 3:4]) * identity[:, 0:1] + raw_out[:, 3:4]
        
        return torch.cat([E_constrained, UTS_constrained, Elong_constrained, Yield_constrained], dim=1)

model = HardConstrainedPINN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-3)

history_train_loss = []
history_val_loss = []

epochs = 2000

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    predictions = model(X_train)
    loss_data = criterion(predictions, Y_train)
    
    loss_data.backward()
    optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_predictions = model(X_val)
        val_loss = criterion(val_predictions, Y_val)
        
    history_train_loss.append(loss_data.item())
    history_val_loss.append(val_loss.item())
    
    if epoch % 400 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {loss_data.item():.5f} | Val Loss: {val_loss.item():.5f}")

model.eval()
with torch.no_grad():
    final_pred = model(X_val)
    y_true = Y_val.cpu().numpy()
    y_pred = final_pred.cpu().numpy()
    final_r2 = r2_score(y_true, y_pred)
    print(f"\nFinal Hard-PINN Val R2 Score: {final_r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(history_train_loss, label='Train Loss')
plt.plot(history_val_loss, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Hard-PINN Training History')
plt.legend()
plt.grid(True)
plt.show()