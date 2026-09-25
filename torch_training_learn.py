import torch
import torch.nn as nn

class DeepMaterialsNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(3, 128)

        self.layer1 = nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, 128)

        self.output_layer = nn.Linear(128, 1)

        self.activation = nn.ReLU()


    def forward(self, X):
        input_layer_res = self.activation(self.input_layer(X))
        layer1_res = self.activation(self.layer1(input_layer_res))
        layer2_res = self.activation(self.layer2(layer1_res))
        output_res = self.output_layer(layer2_res)

        return output_res

model = DeepMaterialsNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# .parameters() - метод для классов, унаследованных от nn.Module

inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

def train(model, optimizer, inputs, targets, epochs):
    for epoch in range(epochs):
        optimizer.zero_grad()  # чистка памяти от старых градиентов (прошлый шаг цикла)

        predictions = model(inputs)

        loss = criterion(predictions, targets)

        loss.backward()
        # после выполнения - в памяти будут лежать посчитанные производные 

        optimizer.step()
        # меняет веса в сети