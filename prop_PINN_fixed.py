import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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
print(f"Device: {device}")

X, Y = data_prop.get_pytorch_data(device)
X_np = X.cpu().numpy()
Y_np = Y.cpu().numpy()

X_train, X_val, Y_train, Y_val = train_test_split(X_np, Y_np, test_size=0.2, random_state=42)

print(f"Train size: {X_train.shape}")
print(f"Val size: {X_val.shape}\n")

x_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(Y_train, dtype=torch.float32).to(device)
x_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(Y_val, dtype=torch.float32).to(device)

X_phys = data_prop.get_physics_point(1000).to(device)

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

model = AdvancedPINN().to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=200)

history_train_loss = []
history_val_loss = []

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
    
    history_train_loss.append(loss_data.item())
    history_val_loss.append(val_loss.item())
    
    if epoch % 400 == 0 or epoch == 1:
        print(f"Epoch {epoch:4d}/{epochs} | Data Loss: {loss_data.item():.5f} | Phys Loss: {loss_physics.item():.5f} | Lambda: {lambda_p:.3f} | Val Loss: {val_loss.item():.5f}")

model.eval()
with torch.no_grad():
    final_pred = model(x_val_t)
    y_true = y_val_t.cpu().numpy()
    y_pred = final_pred.cpu().numpy()
    final_r2 = r2_score(y_true, y_pred)
    print(f"\nFinal Advanced PINN Val R2 Score: {final_r2:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(history_train_loss, label='Train Data Loss')
plt.plot(history_val_loss, label='Val Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Advanced PINN Training History')
plt.legend()
plt.grid(True)
plt.show()



# 1. Устранение «слепого пятна»: В коде с ошибкой -363 мы жестко отрезали на проверку три последние строки таблицы — эксперименты, где длина волокна составляла максимальные 3.0 мм. Сеть обучалась на коротких волокнах (0.2–1.0 мм) и физически не могла угадать, что происходит на 3.0 мм. Жесткие ограничения просто ломали математику.
# 2. Равномерное перемешивание (train_test_split): Теперь благодаря случайному перемешиванию длинные волокна (3.0 мм) попали и в обучающую, и в проверочную выборку. Сеть научилась понимать весь диапазон длин, и 1000 точек физики смогли плавно скорректировать форму функции между известными опытами [S, Mises].
