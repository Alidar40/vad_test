import torch
from torch import nn


class CNN_BiLSTM(nn.Module):
    def __init__(self):
        super(CNN_BiLSTM, self).__init__()

        self.stack = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),

            nn.Linear(1024, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
        )

        self.lstm = nn.LSTM(128, 128, bidirectional=True, batch_first=True)
        self.final = nn.Linear(256, 1)

    def forward(self, x):
        logits = self.stack(x)

        out, hidden = self.lstm(logits)
        out = self.final(out)

        return torch.sigmoid(out)
