# models/mlp.py
import torch.nn as nn
import torch.nn.init as init

class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, activation="relu"):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())

        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU() if activation == "relu" else nn.Tanh())

        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.network(x)