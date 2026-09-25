import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# данные 
X_raw = np.random.randn(100, 3)
Y_raw = np.random.randn(100, 1)

inputs = torch.tensor(X_raw, dtype=torch.float32)
targets = torch.tensor(Y_raw, dtype=torch.float32)
# данные


# модель
class PolymerRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(3, 64)

        self.hidden_layer = nn.Linear(64, 64)

        self.output_layer = nn.Linear(64, 1)

        self.activation = nn.Tanh()

    def forward(self, inputs):
        input_layer_res = self.activation(self.input_layer(inputs))

        hidden_layer_res = self.activation(self.hidden_layer(input_layer_res))

        output_layer_res = self.output_layer(hidden_layer_res)

        return output_layer_res
# модель


model = PolymerRegression()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


def train(model, optimizer, criterion, inputs, targets, epochs=200):
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

    return loss_history


def plot_results(loss_history):
    plt.plot(loss_history, label="Ошибка обучения", color="green")

    plt.xlabel("Эпоха")
    plt.ylabel("Значение ошибки")
    plt.title("График обучения")

    plt.grid(True)
    plt.legend()

    plt.show()


loss_results = train(model, optimizer, criterion=criterion, inputs=inputs, targets=targets)
plot_results(loss_results)