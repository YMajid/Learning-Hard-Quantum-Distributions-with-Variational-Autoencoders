import itertools
from gen_hard import HardStateGenerator
from gen_random import RandomStateGenerator
import numpy as np
from library import Library
import sys
import os
sys.path.insert(-1, "gen_data")


def sample(pdist, n_qubits, N=50000000):
    """
    Samples binary strings according to a probability distribution

    Args:
        pdist: the probability distribution
        n_qubits: number of qubits
        N: number of samples to draw (default is 1000 samples for 50,000 batches worth)
    Returns:
        out: array (N x n_qubits) of binary strings
    Raises:
    """
    bin_nums = [n for n in itertools.product(
        np.array([0, 1]), repeat=n_qubits)]

    ids = np.random.choice(np.arange(len(bin_nums)),
                           size=N, p=pdist, replace=True)
    out = np.take(bin_nums, ids, axis=0)

    return out


def create_dataset(n_qubits=8):
    """
    Generates easy, hard and random datasets and saves them as .npz files

    Args:
        n_qubits: Number of qubits
    Returns:
    Raises:
    """

    np.random.seed(123456789)

    if not os.path.exists('data/'):
        os.makedirs('data/')
    library = Library('data/')

    easy = RandomStateGenerator()
    easy_d = easy.make_dset_easy(n_qubits)
    easy_d = sample(easy_d, n_qubits)
    library.writer(easy_d, 'easy_dataset')
    del easy_d, easy
    print("Finished generating easy dataset.")

    rand = RandomStateGenerator()
    rand_d = rand.make_dset_hard(n_qubits)  # 18qubits is too big to handle
    rand_d = sample(rand_d, n_qubits)
    library.writer(rand_d, 'random_dataset')
    del rand_d, rand
    print("Finished generating random dataset.")

    # 3,4 is the correct inputs for 18 qubit state acc to authors
    hard = HardStateGenerator(3, 4)
    hard_d = hard.get_hard_distribution()
    hard_d = sample(hard_d, n_qubits)
    library.writer(hard_d, 'hard_dataset')
    del hard_d, hard
    print("Finished generating hard dataset.")
