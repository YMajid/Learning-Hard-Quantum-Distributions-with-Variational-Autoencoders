import csv
import sys
sys.path.append("gen_data")
from gen_easy import EasyStateGenerator
from gen_random import RandomStateGenerator
from gen_hard import HardStateGenerator
import numpy as np


def data_reader(csv_path):
    """
    Reads CSV file from given path and returns a features matrix and labels vector.
    Args:
        csv_path: Path to the CSV file to be read.
    Returns:
        features: A nested array of the features from the dataset.
        labels: An array of the labels from the dataset.
    """
    features = []
    labels = []

    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            row_array = list(map(float, row[0].split(' ')))
            features.append(row_array[:-1])
            labels.append(row_array[-1])

    return features, labels

def create_dataset(n_qubits):
    times = np.arange(0, 5.01, 1.0)
    L=4 # mentioned in the paper

    easy = EasyStateGenerator()
    easy_d = easy.get_time_evolve_state(L, times)
    np.savez("easy_dset", easy_dset=easy_d)
    del easy_d, easy
    print("Done easy")

    rand = RandomStateGenerator()
    rand_d = rand.gen_unitary(n_qubits)
    np.savez("rand_dset", rand_dset=rand_d )
    del rand_d, rand
    print("Done random")

    hard = HardStateGenerator(n_qubits, L)
    hard_d = hard.get_hard_distribution(mode="full")
    np.savez("hard_dset", hard_dset=hard_d)
    del hard_d, hard
    print("Done hard")






