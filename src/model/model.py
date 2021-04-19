import torch
import torch.optim as optim
import torch.nn.functional as F
from get_data import get_data
from variational_autoencoder import VariationalAutoencoder
import matplotlib.pyplot as plt
import numpy as np
import os
from hidden_layers import get_layers

class Model:
    def __init__(self, parameters, state='hard', n_layers=3, n_qubits=8):
        self.epochs = int(parameters['epochs'])
        self.batch_size = int(parameters['batch_size'])
        self.display_epochs = int(parameters['display_epoch'])
        self.learning_rate = parameters['learning_rate']
        self.state = state
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.vae, self.train_loaders, self.test_loaders, self.optimizer = self.prepare_model()
        train_losses, test_losses, train_fielities, test_fidelities = self.run_model()
        self.plot_losses(train_losses, test_losses)
        self.plot_fidelities(train_fielities, test_fidelities)

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
        input_size = self.n_qubits #18 if not self.state == 'random' else 15  # should be same as n_qubits I think?
        # n_layers = 2
        VAE_layers = get_layers(input_size, self.n_layers)
        vae = VariationalAutoencoder(VAE_layers.get('encoder'), VAE_layers.get('decoder') , VAE_layers.get('logvar'),  VAE_layers.get('mu')).double().to(self.device)
        train_loaders, test_loaders = get_data(self.batch_size, 'data/', state=self.state)
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
        reconstruction_likelihood = F.binary_cross_entropy(
            x_reconstruction, x, reduction='sum')
        kl_divergence = -0.5 * \
            torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        loss = reconstruction_likelihood + kl_divergence * weight

        return loss

    def fidelity(self, x, x_reconstruction):
        """
        - Calculates the reconstruction fidelity.
        """
        # x = torch.sqrt(x)
        # product = torch.matmul(x, x_reconstruction)
        # product = torch.matmul(product, x)
        # product = torch.sqrt(product)
        # fidelity = torch.trace(product)

        # Bhattacharyya coeff
        out = -torch.log(torch.sum(torch.sqrt(torch.abs(torch.mul(x_reconstruction, x)))))

        return out

        return fidelity



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
        epoch_loss = 0
        fidelity = 0

        # del loader
        # loader = torch.utils.data.DataLoader(np.load("data/easy_dataset.npz")['easy_dset'].astype(float), batch_size=1000, shuffle=True )

        for i, data in enumerate(loader):
            data = data[0].to(self.device)
            self.optimizer.zero_grad()
            reconstruction_data, mu, log_var = self.vae(data)
            loss = self.loss_function(data, reconstruction_data, mu, log_var)
            loss.backward()
            epoch_loss += loss.item() / data.size(0)
            self.optimizer.step()
            fidelity += self.fidelity(data, reconstruction_data).item() / data.size(0)

            if i % 1000 == 0:
                print("Done batch: " + str(i) + "\tCurr Loss: " + str(epoch_loss))

        if (epoch + 1) % self.display_epochs == 0:
            print('Epoch [{}/{}]'.format(epoch + 1, self.epochs) +
                  '\tLoss: {:.4f}'.format(epoch_loss) +
                  '\tFidelity: {:.4f}'.format(fidelity)
                  )

        return epoch_loss, fidelity 

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
        epoch_loss = 0
        fidelity = 0

        with torch.no_grad():
            for i, data in enumerate(loader):
                data = data[0].to(self.device)
                reconstruction_data, mu, logvar = self.vae(data)
                loss = self.loss_function(
                    data, reconstruction_data, mu, logvar)
                epoch_loss += loss.item() / data.size(0)
                fidelity += self.fidelity(data, reconstruction_data).item() / data.size(0)

        return epoch_loss, fidelity

    def run_model(self):
        """
        Args:
            - state: Quantum state the model will be trained on
                - Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """
        # index = 0 if self.state == 'easy' else 1 if self.state == 'hard' else 2
        train_loader, test_loader = self.train_loaders, self.test_loaders
        train_losses, test_losses = [], []
        train_fidelities, test_fidelities = [], []

        print("Beginning Training:")
        for e in range(0, self.epochs):
            train_loss, train_fidelity = self.train(e, train_loader)
            test_loss, test_fidelity = self.test(e, train_loader)
            train_losses.append(train_loss)
            train_fidelities.append(train_fidelity)
            test_losses.append(test_loss)
            test_fidelities.append(test_fidelity)

        torch.save(self.vae.state_dict(), "results/saved_model_{}".format(self.state))

        return train_losses, test_losses, train_fidelities, test_fidelities

    def plot_losses(self, train_losses, test_losses):
        """
        Args:
            - train_losses: list of training losses from run_model
            - test_losses: list of testing losses from run_model
            - state: Quantum state the model was trained on
                - Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """
        epochs = np.arange(0, len(train_losses), 1)
        plt.plot(epochs, train_losses, "g-", label="Training Loss")
        plt.plot(epochs, test_losses, "b-", label="Testing Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Training Loss for the " + str(self.state) + " state")
        plt.legend()
        plt.xlim(0, len(train_losses))
        figure_num = 1
        while os.path.exists(f'results/loss-{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'results/loss-{figure_num}.png')
        print(f'results/loss-{figure_num}.png')

    def plot_fidelities(self, train_fidelities, test_fidelities):
        """
        Args:
            - train_losses: list of training losses from run_model
            - test_losses: list of testing losses from run_model
            - state: Quantum state the model was trained on
                - Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """
        epochs = np.arange(0, len(train_fidelities), 1)
        plt.plot(epochs, train_fidelities, "g-", label="Training Loss")
        plt.plot(epochs, test_fidelities, "b-", label="Testing Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("VAE Training Loss for the " + str(self.state) + " state")
        plt.legend()
        plt.xlim(0, len(test_fidelities))
        figure_num = 1
        while os.path.exists(f'results/fidelities-{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'results/fidelities-{figure_num}.png')
        print(f'results/fidelities-{figure_num}.png')
