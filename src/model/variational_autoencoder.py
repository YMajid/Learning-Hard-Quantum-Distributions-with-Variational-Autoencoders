import torch
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
        self.fc_mu = nn.Linear(10, 4)  # TODO: Fix output layers; should output m
        self.fc_logvar = nn.Linear(10, 4)
        self.LReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        self.encode = nn.Sequential(
            nn.Linear(1000, 800),
            nn.LeakyReLU(0.20),
            nn.Linear(800, 600),
            nn.LeakyReLU(0.20),
            nn.Linear(600, 400),
            nn.LeakyReLU(0.20),
            nn.Linear(400, 200),
            nn.LeakyReLU(0.20),
            nn.Linear(200, 100),
            nn.LeakyReLU(0.20),
            nn.Linear(100, 10),
            nn.Sigmoid()
        )

        self.decode = nn.Sequential(
            nn.Linear(4, 10),
            nn.LeakyReLU(0.20),
            nn.Linear(10, 100),
            nn.LeakyReLU(0.20),
            nn.Linear(100, 200),
            nn.LeakyReLU(0.20),
            nn.Linear(200, 400),
            nn.LeakyReLU(0.20),
            nn.Linear(400, 600),
            nn.LeakyReLU(0.20),
            nn.Linear(600, 800),
            nn.LeakyReLU(0.20),
            nn.Linear(800, 1000),
            nn.Sigmoid()
        )

    def encoder(self, x):
        x = self.encode(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def decoder(self, x):
        x = self.decode(x)
        return x

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        x = self.reparameterize(mu, logvar)
        reconstruction = self.decoder(x)
        return reconstruction, mu, logvar
