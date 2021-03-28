import numpy as np
from library import Library
from gen_easy import EasyStateGenerator
from gen_random import RandomStateGenerator
from gen_hard import HardStateGenerator

library = Library()


def create_dataset(n_qubits=3, L=4, t_i=0.0, t_f=5.01):
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

    hard = HardStateGenerator(n_qubits, L)
    hard_d = hard.get_hard_distribution(mode="full")
    library.writer(hard_d, 'hard_dataset')
    del hard_d, hard
    print("Finished generating hard dataset.")
