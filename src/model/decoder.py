import os
import torch.nn as nn


class Decoder(nn.Module):
    """
    Architecture:
        - Six hidden layers
    """
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(1000, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 150)
        self.fc4 = nn.Linear(150, 150)
        self.fc5 = nn.Linear(150, 150)
        self.fc6 = nn.Linear(150, 10)
