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
    def __init__(self, parameters, state='hard', n_layers=3, n_qubits=8, load=None):
        self.compression = 0.5
        self.epochs = int(parameters['epochs'])
        self.batch_size = int(parameters['batch_size'])
        self.display_epochs = int(parameters['display_epoch'])
        self.learning_rate = parameters['learning_rate']
        self.state = state
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.num_batches = int(parameters['num_batches'])
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.vae, self.train_loaders, self.test_loaders, self.optimizer = self.prepare_model(
            load=load)

        if load == None:
            train_losses, test_losses, train_fielities, test_fidelities = self.run_model()
            self.plot_losses(train_losses, test_losses)
            self.plot_fidelities(train_fielities, test_fidelities)
        else:
            self.fidelity = self.get_fidelity(self.train_loaders)


    def prepare_model(self, load=None):
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
        input_size = self.n_qubits
        VAE_layers = get_layers(input_size, self.n_layers, self.compression)
        vae = VariationalAutoencoder(VAE_layers.get('encoder'), VAE_layers.get(
            'decoder'), VAE_layers.get('logvar'),  VAE_layers.get('mu')).double().to(self.device)
        train_loaders, test_loaders = get_data(
            self.batch_size, 'data/', state=self.state)
        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)

        if not load == None:
            vae.load_state_dict(torch.load(load))
            vae.eval()

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

    def get_fidelity(self, x):
        """
        - Calculates the reconstruction fidelity.
        
        Args:
            -x
        Returns:
            -Fidelity for the input sample
        Raises:
        """
        self.device = torch.device('cpu')
        self.vae.to(self.device)
        x = x.dataset
        x = x.dot(1 << np.arange(x.shape[-1] - 1, -1, -1)) # Converts binary string to integer
        l, u = x.min(), x.max()+1
        f1, b = np.histogram(x, density=True, bins=np.arange(l, u, 1))

        f2 = np.zeros(f1.shape)
        ns = 0
        dim = int(self.n_qubits * self.compression)
        while ns < 10:
            re = np.random.multivariate_normal(
                np.zeros(dim), np.eye(dim), size=int(0.375e7))
            re = self.vae.decode(torch.Tensor(re).double().to(
                self.device)).detach().numpy()
            x_re = re.dot(1 << np.arange(re.shape[-1] - 1, -1, -1))
            f2 += np.histogram(x_re, density=True, bins=b)[0]
            print(f"Sampled fidelity {ns}")
            ns += 1

        out = np.sqrt(np.abs(np.matmul(f1, f2))).sum()
        print(f"Fidelity: {out}")
        del re, x_re, f1, f2, x
        torch.cuda.empty_cache()

        return out

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

        for i, data in enumerate(loader):

            if i >= self.num_batches:
                break

            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction_data, mu, log_var = self.vae(data)
            loss = self.loss_function(
                data, reconstruction_data, mu, log_var, weight=0.85*(epoch/self.epochs))
            loss.backward()
            epoch_loss += loss.item() / (data.size(0) * self.num_batches)
            self.optimizer.step()

            if i % 1000 == 0:
                print("Done batch: " + str(i) +
                      "\tCurr Loss: " + str(epoch_loss))

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

                if i >= self.num_batches:
                    break

                data = data.to(self.device)
                reconstruction_data, mu, logvar = self.vae(data)
                loss = self.loss_function(
                    data, reconstruction_data, mu, logvar)
                epoch_loss += loss.item() / (data.size(0) * self.num_batches)

        return epoch_loss, fidelity

    def run_model(self):
        """
        Args:
            - state: Quantum state the model will be trained on
                - Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """

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

        print(
            f"Final train loss: {train_loss}\tFinal test loss: {test_loss}\tFinal Fidelity: {test_fidelity}")

        torch.save(self.vae.state_dict(),
                   f"results/saved_model_{self.state}_L{self.n_layers}")

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
        plt.title("VAE Training Loss for the " + str(self.state) +
                  " state with " + str(self.n_layers) + "layers")
        plt.legend()
        plt.xlim(0, len(train_losses))
        figure_num = 1
        while os.path.exists(f'results/loss-{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'results/loss-{figure_num}.png')
        plt.clf()
        print(f'results/loss-{figure_num}.png')

    def plot_fidelities(self, fs, state=None):
        """
        Args:
            - fs - A list of Fidelities from each model
            - state: Quantum state the model was trained on
                - Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """

        if state == None:
            state = self.state

        epochs = np.arange(1, len(fs)+1, 1)
        plt.plot(epochs, fs, "b--o", label="Fidelity")
        plt.xlabel("Layers")
        plt.xticks(ticks=epochs)
        plt.ylabel("Fidelity")
        plt.title("VAE Fidelities for the " + str(state) +
                  " state")
        plt.xlim(epochs.min(), epochs.max())
        figure_num = 1
        while os.path.exists(f'results/fidelities-{state}-{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'results/fidelities-{state}-{figure_num}.png')
        plt.clf()
        print(f'results/fidelities-{state}-{figure_num}.png')
