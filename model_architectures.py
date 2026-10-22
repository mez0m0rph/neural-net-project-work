
import torch.nn as nn


class FFNNTwoHidden25(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 25),
            nn.Tanh(),
            nn.Linear(25, 25),
            nn.Tanh(),
            nn.Linear(25, 4)
        )

    def forward(self, x):
        return self.network(x)


class FFNNSingleHidden16(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.Tanh(),
            nn.Linear(16, 4)
        )

    def forward(self, x):
        return self.network(x)


class FFNNSingleHidden8(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
            nn.Linear(8, 4)
        )

    def forward(self, x):
        return self.network(x)