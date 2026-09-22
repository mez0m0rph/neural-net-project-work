import numpy as np
import matplotlib.pyplot as plt

# ОБУЧАЮЩАЯ ВЫБОРКА

np.random.seed(15)

x_min, x_max = -np.pi, np.pi

# кол-во обучающих примеров (пара: вход - правильный ответ)
n_train = 60
# генерируются 40 рандомных чисел из диапазона [-pi,pi] и переводятся в матрицу 40x1
X_train = np.random.uniform(x_min, x_max, n_train).reshape(-1, 1)
# берутся 40 значений из матрицы x_train, для каждого считается синус, перевод в матрицу
y_train = np.sin(X_train) # + np.random.normal(0, 0.1, size=X_train.shape) 

# создает 300 равно расположенных чисел между [-pi, pi] (только для построения графика)
X_plot = np.linspace(x_min, x_max, 300).reshape(-1, 1)
# считает синусы для всех 300 равнорасположенных точек (для эталонной линии на графике)
y_true = np.sin(X_plot)


# СЕТЬ

# функция активации скрытого слоя (просто обертка над готовой функ. NumPy)
def tanh(z):
    return np.tanh(z)

# производная гиперболического тангенса. Для обучения (backpropagation работает через них)
def tanh_derivative(a):
    return 1 - a ** 2

class FeedForwardNet: 
    def __init__(self, n_hidden): # конструктор класса
        self.n_hidden = n_hidden

        # веса первого слоя
        # создает таблицу из 1 строки и n_hidden столбцов со случайными числами
        # по нормальному распределению (??? в основном небольшие числа около нуля ???)
        # умножение на 0.5 - просто уменьшение стартовых значений весов, иначе tanh будет
        # в районе -1/1, где производная почти нулевая, и обучение пойдет плохо
        self.W1 = np.random.randn(1, n_hidden) * 0.5
        #смещение для скрытого слоя, создает матрицу размеров [] заполненную нулями
        self.b1 = np.zeros((1, n_hidden))

        self.W2 = np.random.randn(n_hidden, 1) * 0.5
        # размер матрицы 1x1, потому что на выходе - одно число
        self.b2 = np.zeros((1, 1))

    # метод, который считает, что выдает сеть на конкретном X
    # (сумма весов на входе + bias, активация, матричное умножение)
    def forward(self, x):  # прямой проход
        # взвешенная сумма для скрытого слоя. результат - таблица (n_примеров, n_hidden)
        self.Z1 = x @ self.W1 + self.b1
        # применяем активацию к каждому числу в таблице
        self.A1 = tanh(self.Z1)

        # берем выход скрытого слоя (A1), как "входы" для следующего слоя
        # после - опять подсчет взвешенной суммы
        self.Z2 = self.A1 @ self.W2 + self.b2
        # здесь активация не применяется (стандартный набор для задач регрессии)
        # если пропустить Z2 через tanh, сеть бы максимум выдала число [-1, 1]
        # это подошло бы для синуса, но сломалось бы для других функций 
        # (функций с другим диапазоном значений)
        self.A2 = self.Z2

        return self.A2


# ОБУЧЕНИЕ

# mean squared error
# функция потерь (число, показывающее, насколько сильно сеть ошибается)
# считается ошибка (разница между предсказанием сети и истинным синусом (y_pred, y_true))
# (** 2) - каждая ошибка возводится в квадрат, чтобы избавиться от минусов 
# потому что -0.5 и 0.5 одинаково плохо
# np.mean() - складывает все 40 квадратов ошибок и делит на 40
# получается одно число - средняя квадратичная ошибка (MSE). Цель обучения - свести к нулю
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

def backward(net, X, y, lr):
    n = X.shape[0] # число примеров

    # производная функции потерь по выходу сети 
    # (куда и насколько сильно нужно подвинуть выход, чтобы уменьшить ошибку)
    # net.A2 - что сеть предсказала, y - правильный ответ
    # если величина положительная (надо уменьшить), если отрицательная (надо увеличить)
    dZ2 = (2 / n) * (net.A2 - y)

    # backpropagation ("обратное распространение ошибки")
    dW2 = net.A1.T @ dZ2  # вклад весов w2 в ошибку
    db2 = np.sum(dZ2, axis=0, keepdims=True)  # вклад bias b2 в ошибку

    dA1 = dZ2 @ net.W2.T  # ошибка, пришедшая на скрытый слой

    dZ1 = dA1 * tanh_derivative(net.A1)  # с учетом производной tanh

    dW1 = X.T @ dZ1  # вклад весов w1 в ошибку
    db1 = np.sum(dZ1, axis=0, keepdims=True)  # вклад bias b1 в ошибку

    # шаг градиентного спуска (сдвигаем веса против направления градиента)
    net.W2 -= lr * dW2
    net.b2 -= lr * db2
    net.W1 -= lr * dW1
    net.b1 -= lr * db1

def train(net, X, y, epochs, lr):
    loss_history = []  # список ошибок на каждом шаге (для построения графика)
    for epoch in range(epochs):
        y_pred = net.forward(X)  # прямой проход
        loss = mse_loss(y_pred, y)  # считаем ошибку
        backward(net, X, y, lr)  # обратный проход + обновление весов
        loss_history.append(loss)
    return loss_history


# параметры обучения
n_hidden = 8  # число нейронов скрытого слоя
learning_rate = 0.05
epochs = 5000

net = FeedForwardNet(n_hidden)
loss_history = train(net, X_train, y_train, epochs, learning_rate)



# ГРАФИКИ

y_pred_plot = net.forward(X_plot)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(X_plot, y_true, label='sin(x) (истинная функция)', color='black', 
             linewidth=2)
axes[0].plot(X_plot, y_pred_plot, label='предсказание сети', color='red', linestyle='--')
axes[0].scatter(X_train, y_train, label='обучающие точки', color='blue', s=15, alpha=0.6)
axes[0].set_xlabel('x')
axes[0].set_ylabel('y')
axes[0].set_title(f'Аппроксимация sin(x), скрытый слой = {n_hidden} нейронов')
axes[0].legend()

axes[1].plot(loss_history)
axes[1].set_xlabel('эпоха')
axes[1].set_ylabel('MSE (ошибка)')
axes[1].set_yscale('log')
axes[1].set_title('Кривая обучения')

plt.tight_layout()
plt.savefig('result.png', dpi=150)
plt.show()


# ГРАФИКИ СРАВНЕНИЯ РАЗНОГО ЧИСЛА НЕЙРОНОВ СКРЫТОГО СЛОЯ (2, 4, 6, 8, 10)


hidden_sizes = [2, 4, 6, 8, 10]
results = {}  # n_hidden -> (сеть, история ошибки)
 
for h in hidden_sizes:
    net_h = FeedForwardNet(h)
    loss_h = train(net_h, X_train, y_train, epochs, learning_rate)
    results[h] = (net_h, loss_h)
 
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
 
# график 1: аппроксимация для каждого варианта n_hidden
axes2[0].plot(X_plot, y_true, label='sin(x) (истинная функция)', color='black', linewidth=2)
colors = plt.cm.viridis(np.linspace(0, 1, len(hidden_sizes)))
for (h, (net_h, loss_h)), color in zip(results.items(), colors):
    y_pred_h = net_h.forward(X_plot)
    axes2[0].plot(X_plot, y_pred_h, label=f'{h} нейронов', color=color, linestyle='--')
axes2[0].set_xlabel('x')
axes2[0].set_ylabel('y')
axes2[0].set_title('Аппроксимация при разном числе нейронов')
axes2[0].legend()
 
# график 2: финальная ошибка (MSE) в зависимости от числа нейронов
final_losses = [results[h][1][-1] for h in hidden_sizes]
axes2[1].plot(hidden_sizes, final_losses, marker='o')
axes2[1].set_xlabel('число нейронов скрытого слоя')
axes2[1].set_ylabel('финальная MSE')
axes2[1].set_title('Точность в зависимости от размера скрытого слоя')
 
plt.tight_layout()
plt.savefig('result_comparison.png', dpi=150)
plt.show()