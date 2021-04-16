import numpy as np
from library import Library
import sys
sys.path.insert(-1, "gen_data")
from gen_easy import EasyStateGenerator
from gen_random import RandomStateGenerator
from gen_hard import HardStateGenerator

def binarize(dset):
    """
    - Transforms a probability distribution into binary strings that represent
    basis elements

    Args:
        dset:

    Returns:

    """
    u = np.unique(dset)
    out = np.zeros((dset.shape[0], u.shape[0]))

    for i in range(len(dset)):
        # sets each row to an array of zeros with a 1 corresponding to which unique element it is
        out[i] = (u==dset[i]).astype(float)

    return out


def create_dataset(n_qubits=18, L=2, t_i=0.0, t_f=5.01):
    """
    - Generates easy, hard and random datasets and saves them as .npz files

    Args:
        - n_qubits: Number of qubits
        - L:
        - t_i: Initial time
        - t_f: Final time
    Returns:
    Raises:
    """

    library = Library('data/l{}n{}/'.format(L, n_qubits))
    times = np.arange(t_i, t_f, 1.0)

    easy = EasyStateGenerator()
    easy_d = easy.get_time_evolve_state(L, times)
    library.writer(easy_d, 'easy_dataset')
    del easy_d, easy
    print("Finished generating easy dataset.")

    rand = RandomStateGenerator()
    rand_d = rand.gen_unitary(n_qubits)
    library.writer(rand_d, 'random_dataset')
    del rand_d, rand
    print("Finished generating random dataset.")

    # the n input here is not qubits!!
    # n-qubits = log_2 ( L^n^2 )
    hard = HardStateGenerator(3, 4) # correct inputs for 18 qubit state
    hard_d = hard.get_hard_distribution(mode="full")
    library.writer(hard_d, 'hard_dataset')
    del hard_d, hard
    print("Finished generating hard dataset.")
