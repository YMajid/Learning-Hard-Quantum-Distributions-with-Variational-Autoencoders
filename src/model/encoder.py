import os
import torch.nn as nn


class Encoder(nn.Module):
    """
    Architecture:
        - Six hidden layers
        - Leaky rectified linear units function
        - Sigmoid activation function
    """
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(1000, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150, 150)
        self.fc6 = nn.Linear(150, 10)
        self.LReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def reset(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()
        self.fc4.reset_parameters()
        self.fc5.reset_parameters()
        self.fc6.reset_parameters()
        return

    def forward(self, x):
        x = self.fc1(x)
        x = self.LReLU(x)
        x = self.fc2(x)
        x = self.LReLU(x)
        x = self.fc3(x)
        x = self.LReLU(x)
        x = self.fc4(x)
        x = self.LReLU(x)
        x = self.fc5(x)
        x = self.LReLU(x)
        x = self.fc6(x)
        x = self.LReLU(x)
        x = self.sigmoid(x)
        return x
