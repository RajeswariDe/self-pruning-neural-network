import torch
import torch.nn as nn
import torch.nn.functional as F

class PrunableLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )

        self.bias = nn.Parameter(
            torch.zeros(out_features)
        )

        self.gate_scores = nn.Parameter(
            torch.randn(out_features, in_features)
        )

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)

        pruned_weight = self.weight * gates

        return F.linear(x, pruned_weight, self.bias)

    def get_gates(self):
        return torch.sigmoid(self.gate_scores)


class PrunableMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),

            PrunableLinear(3 * 32 * 32, 512),
            nn.ReLU(),

            PrunableLinear(512, 256),
            nn.ReLU(),

            PrunableLinear(256, 10)
        )

    def forward(self, x):
        return self.net(x)