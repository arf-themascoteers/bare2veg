import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(7,10),
            nn.LeakyReLU(),
            nn.Linear(10, 10),
            nn.LeakyReLU(),
            nn.Linear(10, 7),
            nn.Tanh()
        )

    def forward(self, x):
        return self.linear(x)