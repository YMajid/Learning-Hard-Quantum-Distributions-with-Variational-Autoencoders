import numpy as np
from library import Library
import sys
sys.path.insert(-1, "gen_data")
from gen_easy import EasyStateGenerator
from gen_random import RandomStateGenerator
from gen_hard import HardStateGenerator
import itertools

def sample(pdist, n_qubits, N=50000000):
    """
    - Samples binary strings according to a probability distribution

    Args:
        pdist: the probability distribution
        n_qubits: number of qubits
        N: number of samples to draw (default is 1000 samples for 50,000 batches worth)

    Returns: array(N x n_qubits) of binary strings

    """
    bin_nums = [n for n in itertools.product(np.array([0,1]), repeat=n_qubits)]

    ids = np.random.choice(np.arange(len(bin_nums)), size=N, p=pdist, replace=True )
    out = np.take(bin_nums, ids, axis=0)

    return out

def create_dataset(n_qubits=8, L=2, t_i=0.0, t_f=5.01):
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

    # library = Library('data/l{}n{}/'.format(L, n_qubits))
    library = Library('data/')
    # times = np.arange(t_i, t_f, 1.0)

    # easy = EasyStateGenerator()
    # easy_d = easy.get_time_evolve_state(L, times)
    # library.writer(easy_d, 'easy_dataset')
    # del easy_d, easy
    # print("Finished generating easy dataset.")

    easy = RandomStateGenerator()
    easy_d = easy.make_dset_easy(n_qubits)
    easy_d = sample(easy_d, n_qubits)
    library.writer(easy_d, 'easy_dataset')
    del easy_d, easy
    print("Finished generating easy dataset.")

    rand = RandomStateGenerator()
    rand_d = rand.make_dset_hard(n_qubits) # 18qubits is too big to handle
    rand_d = sample(rand_d, n_qubits)
    library.writer(rand_d, 'random_dataset')
    del rand_d, rand
    print("Finished generating random dataset.")

    # the n input here is not qubits!!
    # n-qubits = log_2 ( L^n^2 )
    hard = HardStateGenerator(3, 4) # 3,4 is the correct inputs for 18 qubit state
    hard_d = hard.get_hard_distribution(mode="full")
    hard_d = sample(hard_d, n_qubits)
    library.writer(hard_d, 'hard_dataset')
    del hard_d, hard
    print("Finished generating hard dataset.")
