# Imports
import sys
import json
import argparse
import os

sys.path.append('src/model')
sys.path.append('src/utils')
sys.path.append("src/utils/gen_data")

from create_dataset import create_dataset

from model import Model

if __name__ == '__main__':
    print("Main File.")

    # Read parameter JSON file, convert it into a Python dictionary
    with open('param/parameters.json') as f:
        parameters = json.loads(f.read())
        f.close()

    n = 18 # number of qubits

    # create dataset if not available locally (only takes a minute or three)
    if not os.path.exists('data/easy_dataset.npz'):
        print("Creating dataset, please wait one moment")
        create_dataset(n_qubits=n)
    else:
        print("Dataset found")

    # Generate, train, and save each model
    # for state in ['easy', 'random', 'hard']:
    #     for i in range(1,6):
    #         m = Model(parameters, state=state, n_qubits=n, n_layers=i)

    # Load and plot fidelities
    for state in ['easy', 'random', 'hard']:
        fs = []
        for i in range(1,6):
            m = Model(parameters, state=state, n_qubits=n, n_layers=i, load=f"results/saved_model_{state}_L{i}")
            fs.append(m.fidelity)
        m.plot_fidelities(fs, state=state)