import torch.nn as nn


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder class.

    Architecture:
        - 6 Fully connected layers
        - Sigmoid activation function
        - LeakyReLU activation function with slope of -0.2
    """

    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.fc1 = nn.Linear(1000, 800)
        self.fc2 = nn.Linear(800, 600)
        self.fc3 = nn.Linear(600, 400)
        self.fc4 = nn.Linear(400, 200)
        self.fc5 = nn.Linear(200, 100)
        self.fc6 = nn.Linear(100, 10)
        self.LReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        pass

    def decode(self, x):
        pass

    def reparameterize(self):
        pass

    def forward(self, x):
        pass
