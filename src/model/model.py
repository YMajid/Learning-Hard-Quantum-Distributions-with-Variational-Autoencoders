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
        self.vae, self.train_loaders, self.test_loaders, self.optimizer = self.prepare_model(load=load)

        if load == None:
            train_losses, test_losses, train_fielities, test_fidelities = self.run_model()
            self.plot_losses(train_losses, test_losses)
            self.plot_fidelities(train_fielities, test_fidelities)
        else:
            f = self.fidelity(self.train_loaders)
            print(f)

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
        input_size = self.n_qubits #18 if not self.state == 'random' else 15  # should be same as n_qubits I think?
        # n_layers = 2
        VAE_layers = get_layers(input_size, self.n_layers)
        vae = VariationalAutoencoder(VAE_layers.get('encoder'), VAE_layers.get('decoder') , VAE_layers.get('logvar'),  VAE_layers.get('mu')).double().to(self.device)
        train_loaders, test_loaders = get_data(self.batch_size, 'data/', state=self.state)
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

    def fidelity(self, x):
        """
        - Calculates the reconstruction fidelity.
        """
        # old
        # x = torch.sqrt(x)
        # product = torch.matmul(x, x_reconstruction)
        # product = torch.matmul(product, x)
        # product = torch.sqrt(product)
        # fidelity = torch.trace(product)

        # newer
        # Convert x, x_reconsruction to probability distributions
        # x_uniques, x_freqs = np.unique(x.cpu().detach().numpy(), return_counts=True, axis=0)
        # x_uniques_re, x_freqs_re = np.unique(x_reconstruction.cpu().detach().numpy(), return_counts=True, axis=0)
        #
        # x_freqs_re = np.divide(x_freqs_re, x_reconstruction.size(1))
        # x_freqs = np.divide(x_freqs, x.size(1))


        # Bhattacharyya coeff
        # out = torch.sum(torch.sqrt(torch.abs(torch.mul(x_freqs_re, x_freqs))))
        # out = np.sqrt(np.abs(np.matmul(x_freqs, x_freqs_re))).sum()

        # return out

        self.vae.to(torch.device('cpu'))
        re = self.vae(torch.Tensor(x.dataset).double())
        re = re[0].detach().numpy()
        x = x.dataset
        x = x.dot(1 << np.arange(x.shape[-1] - 1, -1, -1))
        x_re = re.dot(1 << np.arange(re.shape[-1] - 1, -1, -1))

        l, u = x.min(), x.max()
        f1, b = np.histogram(x, density=True, bins=np.arange(l,u,1))
        f2, _ = np.histogram(x_re, density=True, bins=b)

        plt.hist(x, bins=b, density=True )
        plt.savefig('og.png', dpi=500)
        plt.clf()
        plt.close()

        plt.hist(x_re, bins=b, density=True )
        plt.savefig("re.png", dpi=500)
        plt.clf()
        plt.close()

        print(l,u)
        print(b)

        out = np.sqrt(np.abs(np.matmul(x, x_re))).sum()
        print(out)
        print(out/ x.shape[0])
        self.vae.to(torch.device('cuda'))

        return out / x.shape[0]


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

            if i >= self.num_batches: break

            data = data.to(self.device)
            self.optimizer.zero_grad()
            reconstruction_data, mu, log_var = self.vae(data)
            loss = self.loss_function(data, reconstruction_data, mu, log_var)
            loss.backward()
            epoch_loss += loss.item() / (data.size(0) * self.num_batches )
            self.optimizer.step()


            if i % 1000 == 0:
                print("Done batch: " + str(i) + "\tCurr Loss: " + str(epoch_loss))

        # fidelity = self.fidelity(loader)
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

                if i >= self.num_batches: break

                data = data.to(self.device)
                reconstruction_data, mu, logvar = self.vae(data)
                loss = self.loss_function(
                    data, reconstruction_data, mu, logvar)
                epoch_loss += loss.item() /(data.size(0) * self.num_batches )
            # fidelity = self.fidelity(loader)

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

        print(f"Final train loss: {train_loss}\tFinal test loss: {test_loss}\tFinal Fidelity: {test_fidelity}")

        torch.save(self.vae.state_dict(), f"results/saved_model_{self.state}_L{self.n_layers}")

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
        plt.title("VAE Training Loss for the " + str(self.state) + " state with " + str(self.n_layers) + "layers")
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
        plt.title("VAE Training Loss for the " + str(self.state) + " state with " + str(self.n_layers) + "layers")
        plt.legend()
        plt.xlim(0, len(test_fidelities))
        figure_num = 1
        while os.path.exists(f'results/fidelities-{figure_num}.png'):
            figure_num += 1
        plt.savefig(f'results/fidelities-{figure_num}.png')
        print(f'results/fidelities-{figure_num}.png')
