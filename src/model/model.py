import torch
import torch.optim as optim
import torch.nn.functional as F
from get_data import get_data
from variational_autoencoder import VariationalAutoencoder


class Model:
    def __init__(self, parameters):
        self.epochs = int(parameters['epochs'])
        self.batch_size = int(parameters['batch_size'])
        self.display_epochs = int(parameters['display_epoch'])
        self.learning_rate = parameters['learning_rate']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vae, self.train_loaders, self.test_loaders, self.optimizer = self.prepare_model()
        self.run_model()

    def prepare_model(self):
        """
        - Initializes VAE model and loads it onto the appropriate device.
        - Reads and loads the data in the form of an array of Torch DataLoaders.
            - 0: Easy Quantum State
            - 1: Hard Quantum State
            - 2: Random Quantum State
        - Initializes Adam optimizer.

        Args:
        Returns:
            - VAE
            - Array of train Torch Dataloaders
            - Array of test Torch Dataloaders
            - Adam optimizer
        Raises:
        """
        vae = VariationalAutoencoder().double().to(self.device)

        train_loaders, test_loaders = get_data(self.batch_size, 'data/l2n4/')

        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)

        return vae, train_loaders, test_loaders, optimizer

    def loss_function(self, x, x_reconstruction, mu, log_var, weight=1):
        """
        - Returns the loss for the model based on the reconstruction likelihood and KL divergence

        Args:
            - x
            - x_reconstruction
            - mu
            - log_var
        Returns:
            - Model loss
        Raises:
        """
        reconstruction_likelihood = F.binary_cross_entropy(x_reconstruction, x.view(-1), reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconstruction_likelihood + kl_divergence * weight

        return loss

    def train(self, epoch, loader):
        """
        - Trains the VAE model

        Args:
            - epoch: Number of current epoch to print
            - loader: Torch DataLoader for a quantum state
        Returns:
        Raises:
        """
        self.vae.train()

        for i, data in enumerate(loader):
            data = data[0].to(self.device)

            self.optimizer.zero_grad()
            reconstruction_batch, mu, log_var = self.vae(data)
            loss = self.loss_function(data, reconstruction_batch, mu, log_var)
            loss.backward()
            self.optimizer.step()

            if epoch + 1 % self.display_epochs == 0:
                print('Epoch [{}/{}]'.format(epoch + 1, self.epochs) + \
                      '\tLoss: {:.4f}'.format(loss.item()))

    def test(self, epoch, loader):
        """
        - Tests VAE model

        Args:
            - epoch: Number of current epoch to print
            - loader: Torch DataLoader for a quantum state
        Returns:
        Raises:
        """
        self.vae.eval()

        with torch.no_grad():
            for i, (data, _) in enumerate(loader):
                pass

    def run_model(self, state='hard'):
        """
        Args:
            - state: Quantum state the model will be trained on
                - Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """
        index = 0 if state == 'easy' else 1 if state == 'hard' else 2
        train_loader, test_loader = self.train_loaders[index], self.test_loaders[index]

        for e in range(0, self.epochs):
            self.train(e, train_loader)
            # self.test(e, test_loader)
