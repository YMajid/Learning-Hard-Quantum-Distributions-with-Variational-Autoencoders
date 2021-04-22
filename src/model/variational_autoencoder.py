import torch
import torch.nn as nn

# Useful for debugging, just prints the shape of the input
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

    def __init__(self, encode, decode, logvar, mu):
        """
        Very standard VAE, all the heavy lifting done elsewhere
        Args:
            encode: encoder input from hidden_layers
            decode: decoder layers
            logvar: logvar layer
            mu:mu layer
        """

        super(VariationalAutoencoder, self).__init__()
        self.LReLU = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()

        self.fc_logvar = logvar
        self.fc_mu = mu

        self.encode = encode
        self.decode = decode

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
