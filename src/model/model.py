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
    def __init__(self, parameters, verbosity=0, state='hard', n_layers=3, n_qubits=8, load=None):
        """
        Args:
            parameters: dict of json params
            state: quantum state: 'easy', 'random', or 'hard
            n_layers: number of layers in the encoder/decoder
            n_qubits: number of qubits
            load: optional path to load a pretrained model
        """
        # Initialize class parameteres
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
        self.verbosity = verbosity

        # Prepare model
        self.vae, self.train_loaders, self.test_loaders, self.optimizer = self.prepare_model(
            load=load)

        # Train the model if it wasn't loaded, and compute fidelity
        if load == None:
            train_losses, test_losses = self.run_model()
            self.plot_losses(train_losses, test_losses)
        self.fidelity = self.get_fidelity(self.train_loaders)

    def prepare_model(self, load=None):
        """
        Initializes VAE model and loads it onto the appropriate device.
        Reads and loads the data in the form of an array of Torch DataLoaders.
        Initializes Adam optimizer.

        Args:
            load: path to load trained model from
        Returns:
            VAE
            Array of train Torch Dataloaders
            Array of test Torch Dataloaders
            Adam optimizer
        Raises:
        """
        input_size = self.n_qubits
        VAE_layers = get_layers(input_size, self.n_layers, self.compression)
        vae = VariationalAutoencoder(VAE_layers.get('encoder'), VAE_layers.get(
            'decoder'), VAE_layers.get('logvar'), VAE_layers.get('mu')).double().to(self.device)
        train_loaders, test_loaders = get_data(
            self.batch_size, 'data/', state=self.state)
        optimizer = optim.Adam(vae.parameters(), lr=self.learning_rate)

        if not load == None:
            vae.load_state_dict(torch.load(load))
            vae.eval()

        return vae, train_loaders, test_loaders, optimizer

    def loss_function(self, x, x_reconstruction, mu, log_var, weight=1):
        """
        Returns the loss for the model based on the reconstruction likelihood and KL divergence

        Args:
            x: Input data
            x_reconstruction: Reconstructed data
            mu:
            log_var:
            weight:
        Returns:
            loss:
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
        Calculates the reconstruction fidelity.
        
        Args:
            x: Input data
        Returns:
            out: Fidelity for the input sample
        Raises:
        """

        # Get whole dataset, convert to integer, get bounds, compute (true) probability density
        x = x.dataset
        x = x.dot(1 << np.arange(x.shape[-1] - 1, -1, -1))  # Converts binary string to integer
        l, u = x.min(), x.max() + 1
        f1, b = np.histogram(x, density=True, bins=np.arange(l, u, 1))

        # Initialize for getting reconstructed density
        f2 = np.zeros(f1.shape)
        ns = 0
        dim = int(self.n_qubits * self.compression)
        while ns < 10:
            # Get samples, decode them, convert to int, and add to hist count
            re = np.random.multivariate_normal(
                np.zeros(dim), np.eye(dim), size=int(0.375e7))
            re = self.vae.decode(torch.Tensor(re).double().to(
                self.device)).cpu().detach().numpy()
            x_re = re.dot(1 << np.arange(re.shape[-1] - 1, -1, -1))
            f2 += np.histogram(x_re, bins=b)[0]
            print(f"Sampled fidelity {ns}")
            ns += 1

        # Normalize to density (taken from np.histogram source)
        db = np.array(np.diff(b), float)
        f2 = f2 / db / f2.sum()

        out = np.sum(np.sqrt(np.multiply(f1, f2)))
        print(f"Fidelity: {out}")
        del re, x_re, f1, f2, x
        torch.cuda.empty_cache()

        return out

    def train(self, epoch, loader):
        """
        Trains the VAE model

        Args:
            epoch: Number of current epoch to print
            loader: Torch DataLoader for a quantum state
        Returns:
            epoch_loss: Loss for the epoch
        Raises:
        """
        self.vae.train()
        epoch_loss = 0

        for i, data in enumerate(loader):

            if i >= self.num_batches:
                break

            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction_data, mu, log_var = self.vae(data)
            loss = self.loss_function(
                data, reconstruction_data, mu, log_var, weight=0.85 * (epoch / self.epochs))
            loss.backward()
            epoch_loss += loss.item() / (data.size(0) * self.num_batches)
            self.optimizer.step()

            if (self.verbosity == 0 or (
                    self.verbosity == 1 and (epoch + 1) % self.display_epochs == 0)) and i % self.batch_size == 0:
                print("Done batch: " + str(i) +
                      "\tCurr Loss: " + str(epoch_loss))

        if self.verbosity == 0 or (self.verbosity == 1 and (epoch + 1) % self.display_epochs == 0):
            print('Epoch [{}/{}]'.format(epoch + 1, self.epochs) +
                  '\tLoss: {:.4f}'.format(epoch_loss)
                  )

        return epoch_loss

    def test(self, epoch, loader):
        """
        Tests VAE model

        Args:
            epoch: Number of current epoch to print
            loader: Torch DataLoader for a quantum state
        Returns:
            epoch_loss: Loss for the epoch
        Raises:
        """
        self.vae.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i, data in enumerate(loader):

                if i >= self.num_batches:
                    break

                data = data.to(self.device)
                reconstruction_data, mu, logvar = self.vae(data)
                loss = self.loss_function(
                    data, reconstruction_data, mu, logvar)
                epoch_loss += loss.item() / (data.size(0) * self.num_batches)

        return epoch_loss

    def run_model(self):
        """
        Args:
            state: Quantum state the model will be trained on
                Options include: 'easy', 'hard', 'random'
        Returns:
            test and training loss
        Raises:
        """

        train_loader, test_loader = self.train_loaders, self.test_loaders
        train_losses, test_losses = [], []

        print("Beginning Training:")
        for e in range(0, self.epochs):
            train_loss = self.train(e, train_loader)
            test_loss = self.test(e, train_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)

        print(
            f"Final train loss: {train_loss}\tFinal test loss: {test_loss}")

        torch.save(self.vae.state_dict(),
                   f"results/saved_model_{self.state}_L{self.n_layers}")

        return train_losses, test_losses

    def plot_losses(self, train_losses, test_losses):
        """
        Args:
            train_losses: list of training losses from run_model
            test_losses: list of testing losses from run_model
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
            fs: A list of Fidelities from each model
            state: Quantum state the model was trained on
                Options include: 'easy', 'hard', 'random'
        Returns:
        Raises:
        """

        if state == None:
            state = self.state

        epochs = np.arange(1, len(fs) + 1, 1)
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
