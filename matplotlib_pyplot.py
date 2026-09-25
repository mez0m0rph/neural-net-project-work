import matplotlib.pyplot as plt 
import numpy as np

epochs = [1, 2, 3, 4, 5]
loss_values = [0.5, 0.3, 0.15, 0.08, 0.02]

plt.plot(epochs, loss_values, label="ошибка обучения", color="red")

plt.xlabel("эпоха")  # текст на оси X
plt.ylabel("значение Loss")  # текст на оси Y
plt.title("График обучения нейросети")  # заголовок

plt.grid(True)  # отображение сетки
plt.legend()  # информативное окно, отображает информацию про график (label)

plt.show()  # рендер итогового изображения

def plot_loss_curve(loss_history):
    plt.plot(loss_history, label="Train Loss", color="blue")  # принимает объекты для отрисовки

    plt.xlabel("эпоха")
    plt.ylabel("MSE Loss")
    plt.title("Процесс обучения модели полимеров")

    plt.grid(True)
    plt.legend()

    plt.show()

loss_history_test = np.arange(10)

plot_loss_curve(loss_history_test)