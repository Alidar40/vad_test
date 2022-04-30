import torch
from torch import nn


class LeNet8(nn.Module):
    def __init__(self):
        super(LeNet8, self).__init__()

        self.lenet_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 8*8
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 64, kernel_size=2, padding=1),  # 4*4
            nn.ReLU(),
            nn.Dropout2d(p=0.3),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 2*2

            nn.Flatten(),

            nn.Linear(64 * 2 * 2, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.lenet_stack(x)
        return torch.sigmoid(logits)


class LeNet32(nn.Module):
    def __init__(self):
        super(LeNet32, self).__init__()

        self.lenet_stack = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),  # 32*32
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=5, padding=2),  # 16*16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 8*8

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 8*8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 4*4

            nn.Conv2d(64, 64, kernel_size=3, padding=1),  # 4*4
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(64 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 1),
        )

    def forward(self, x):
        logits = self.lenet_stack(x)
        return torch.sigmoid(logits)
