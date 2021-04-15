import torch
import torch.nn as nn


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


class VariationalAutoencoder(nn.Module):
    """
    Variational Autoencoder class.

    Architecture:
        - 6 Fully connected layers
        - Sigmoid activation function
        - LeakyReLU activation function with slope of -0.2
    """

    def __init__(self, input_size):
        super(VariationalAutoencoder, self).__init__()
        self.LReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
        self.compr = input_size // 2

        self.reduction = self.compr // 7

        self.fc_logvar = nn.Linear(input_size - self.reduction * 6, self.compr)
        self.fc_mu = nn.Linear(input_size - self.reduction * 6, self.compr)

        self.encode = nn.Sequential(
            nn.Linear(input_size, input_size - self.reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction,
                      input_size - self.reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 2,
                      input_size - self.reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 3,
                      input_size - self.reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 4,
                      input_size - self.reduction * 5),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 5,
                      input_size - self.reduction * 6),
            nn.Sigmoid()
        )
        self.decode = nn.Sequential(
            nn.Linear(self.compr, input_size - self.reduction * 6),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 6,
                      input_size - self.reduction * 5),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 5,
                      input_size - self.reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 4,
                      input_size - self.reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 3,
                      input_size - self.reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 2,
                      input_size - self.reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - self.reduction * 1, input_size),
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

    def division_factors(self):
        pass
