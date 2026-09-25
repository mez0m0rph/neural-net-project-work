import torch
import torch.nn as nn 

# любая нейросеть в pytorch - это класс, наследующийся от nn.Module:
# в нем всегда есть [конструктор __init__(self)] и [метод прямого прохода def forward(self, x)]

# __init__(self) - объявление слои (компоненты сети)
# forward - описание логики движения данных X через слои и функции активации


class MaterialNet(nn.Module):
    def __init__(self):
        super(MaterialNet, self).__init__()  # в python3 не нужно передавать параметры в super() - пишется просто super().__init__()

        self.layer1 = nn.Linear(in_features=2, out_features=64) 

        self.layer2 = nn.Linear(in_features=64, out_features=64)

        self.output_layer = nn.Linear(in_features=64, out_features=2)

        self.activation = nn.Tanh()


    def forward(self, x):
        # пропускаем данные через первый слой и функцию активации
        x = self.activation(self.layer1(x)) 

        x = self.activation(self.layer2(x))

        x = self.output_layer(x)

        return x


class DeepMaterialsNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Linear(3, 128)

        self.layer1 =  nn.Linear(128, 128)
        self.layer2 = nn.Linear(128, 128)

        self.output_layer = nn.Linear(128, 1)

        self.activation = nn.ReLU()


    def forward(self, X):
        input_layer_res = self.activation(self.input_layer(X))
        layer1_res = self.activation(self.layer1(input_layer_res))
        layer2_res = self.activation(self.layer2(layer1_res))
        output_layer_res = self.output_layer(layer2_res)

        return output_layer_res
