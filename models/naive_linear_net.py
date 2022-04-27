import torch
from torch import nn


class NaiveLinearNet(nn.Module):
    def __init__(self):
        super(NaiveLinearNet, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(160, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return torch.sigmoid(logits)
