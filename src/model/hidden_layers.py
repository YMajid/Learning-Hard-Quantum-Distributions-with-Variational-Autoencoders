import torch.nn as nn


def get_layers(input_size, n_layers):

    compression = input_size // 2
    reduction = compression // n_layers

    encoder = None

    if n_layers == 1:
        encoder = nn.Sequential(nn.Sigmoid())

    elif n_layers == 2:
        encoder = nn.Sequential(
                nn.Linear(input_size, input_size - reduction),
                nn.Sigmoid()
                )

    elif n_layers == 3:
        encoder = nn.Sequential(
                nn.Linear(input_size, input_size - reduction),
                nn.LeakyReLU(0.20),
                nn.Linear(input_size - reduction,
                    input_size - reduction * 2),
                nn.Sigmoid()
                )

    elif n_layers == 4:
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
        decoder = nn.Sequential(
                nn.Linear(compression, input_size - reduction * (n_layers - 1)),
                nn.LeakyReLU(0.20),
                nn.Linear(input_size - reduction * 1, input_size),
                nn.Sigmoid()
                )

    elif n_layers == 1:
        decoder = nn.Sequential(
                nn.Linear(input_size - reduction * 1, input_size),
                nn.Sigmoid()
                )

    fc_logvar = nn.Linear(input_size - reduction * (n_layers - 1), compression)
    fc_mu = nn.Linear(input_size - reduction * (n_layers - 1), compression)

    return {'decoder': decoder, 'encoder': encoder, 'logvar': fc_logvar, 'mu': fc_mu}
