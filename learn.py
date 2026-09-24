import torch
import torch.nn as nn 

# любая нейросеть в pytorch - это класс, наследующийся от nn.Module:
# в нем всегда есть [конструктор __init__(self)] и [метод прямого прохода def forward(self, x)]

# __init__(self) - объявление слои (компоненты сети)
# forward - описание логики движения данных X через слои и функции активации


class MaterialNet(nn.Module):
    def __init__(self):
        super(MaterialNet, self).__init__()

        self.layer1 = nn.Linear(in_features=2, out_features=64) 

        self.layer2 = nn.Linear(in_features=64, out_features=64)

        self.output_layer == nn.Linear(in_features=62, out_features=2)

        self.activation = nn.Tanh()


    def forward(self, x):
        # пропускаем данные через первый слой и функцию активации
        x = self.activation(self.layer1(x)) 

        x = self.activation(self.layer2(x))

        x = self.output_layer(x)

        return x