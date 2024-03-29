import torch.nn as nn


class Print(nn.Module):
    """
    Used to print the shape of the data inside of nn.Sequential
    """

    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x


def get_layers(input_size, n_layers, compression):
    """
    Retrieves the architecture of the VAE based on compression

    Args:
        input_size: size of input to first layer
        n_layers: number of desired layers
        compression: desired compression factor

    Returns: dictionary containing encder, decoder, mu, and logvar layers

    """
    print(f'Creating {n_layers} layers')

    # Determine reduction factor
    compression = int(input_size * compression)
    reduction = compression // n_layers

    encoder = None
    # These if statements determine the structure of the VAE dependent on the desired compression

    if n_layers == 1:
        # Desired structure for 1 total layer in the encoder
        encoder = nn.Sequential(nn.Sigmoid())
    elif n_layers == 2:
        # Desired structure for 2 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.Sigmoid()
        )
    elif n_layers == 3:
        # Desired structure for 3 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.Sigmoid()
        )
    elif n_layers == 4:
        # Desired structure for 4 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.Sigmoid()
        )
    elif n_layers == 5:
        # Desired structure for 5 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 4),
            nn.Sigmoid()
        )
    elif n_layers == 6:
        # Desired structure for 6 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 5),
            nn.Sigmoid()
        )
    elif n_layers == 7:
        # Desired structure for 7 total layers in the encoder
        encoder = nn.Sequential(
            nn.Linear(input_size, input_size - reduction),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 5),
            nn.Linear(input_size - reduction * 5,
                      input_size - reduction * 6),
            nn.Sigmoid()
        )

    decoder = None

    if n_layers == 7:
        # Desired structure for 7 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 6,
                      input_size - reduction * 5),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 5,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif (n_layers == 6):
        # Desired structure for 6 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 5,
                      input_size - reduction * 4),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif (n_layers == 5):
        # Desired structure for 5 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 4,
                      input_size - reduction * 3),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 4:
        # Desired structure for 4 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 3,
                      input_size - reduction * 2),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 3:
        # Desired structure for 3 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 2,
                      input_size - reduction * 1),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 2:
        # Desired structure for 2 total layers in the decoder
        decoder = nn.Sequential(
            nn.Linear(compression, input_size - reduction * (n_layers - 1)),
            nn.LeakyReLU(0.20),
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )
    elif n_layers == 1:
        # Desired structure for 1 total layer in the decoder
        decoder = nn.Sequential(
            nn.Linear(input_size - reduction * 1, input_size),
            nn.Sigmoid()
        )

    # Latent log variance and mu layers
    fc_logvar = nn.Linear(input_size - reduction * (n_layers - 1), compression)
    fc_mu = nn.Linear(input_size - reduction * (n_layers - 1), compression)

    return {'decoder': decoder, 'encoder': encoder, 'logvar': fc_logvar, 'mu': fc_mu}
