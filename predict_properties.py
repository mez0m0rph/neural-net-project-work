import torch
import numpy as np
from kan import KAN
import prop_data_2_lg as data_prop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Инициализируем лучшую архитектуру PIKAN (34 скрытых нейрона, grid=4)
model = KAN(width=[2, 34, 4], grid=4, k=3, device=str(device))
model.load_state_dict(torch.load('./best_model/best_pikan_model.pth', map_location=device))
model.eval()

# ==========================================
# ВВЕДИТЕ СЮДА ЛЮБЫЕ ВАШИ ПАРАМЕТРЫ КОМПОЗИТА
# ==========================================
target_cf = 23.5       # Содержание углеволокна в % (произвольное)
target_length = 1.45   # Длина волокна в мм (произвольная)
# ==========================================

input_raw = np.array([[target_cf, target_length]], dtype=np.float32)
input_scaled = data_prop.scaler_X.transform(input_raw)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32).to(device)

with torch.no_grad():
    prediction_scaled = model(input_tensor).cpu().numpy()

# Восстанавливаем оригинальные масштабы свойств
Y_log = data_prop.Y_raw.copy()
Y_log[:, 2] = np.log10(Y_log[:, 2])
y_mins = Y_log.min(axis=0)
y_maxs = Y_log.max(axis=0)

prediction_real = np.zeros_like(prediction_scaled)
for col in range(4):
    prediction_real[:, col] = prediction_scaled[:, col] * (y_maxs[col] - y_mins[col]) + y_mins[col]

# Обратное потенцирование для Elongation (столбец с индексом 2)
prediction_real[:, 2] = 10 ** prediction_real[:, 2]

property_names = ['Young Modulus', 'UTS', 'Elongation', 'Yield Strength']
units = ['GPa', 'MPa', '%', 'MPa']

print("="*60)
print(f"PREDICTED MECHANICAL PROPERTIES FOR: CF = {target_cf}%, Length = {target_length} mm")
print("="*60)
for i in range(4):
    print(f"{property_names[i]:<15}: {prediction_real[0, i]:>7.2f} {units[i]}")
print("="*60)
