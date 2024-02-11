import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.linear = nn.Sequential(
            nn.Linear(14, 5),
            nn.LeakyReLU(),
            nn.Linear(5, 1)
        )

    def forward(self, input, target):
        input_target = torch.cat((input, target), 1)
        return self.linear(input_target)

