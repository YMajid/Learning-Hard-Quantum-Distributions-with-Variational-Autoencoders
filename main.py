# Imports
import sys
import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src/model')
sys.path.append('src/utils')
sys.path.append("src/utils/gen_data")

from create_dataset import create_dataset

from model import Model

def generate_figure(states):
        for state in states:
            tests = 1
            fs = []
            fs_std = []
            for i in range(1,6):
                temp_fs = []
                for _ in range(tests):
                    m = Model(parameters, state=state, n_qubits=18, n_layers=i, load=f"results/saved_model_{state}_L{i}")
                    temp_fs.append(m.fidelity)
                fs.append(np.average(temp_fs))
                fs_std.append(temp_fs)
                
            fs_std = np.std(fs_std)
            m.plot_fidelities(fs, fs_std, state=state)

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
    
    generate_figure(['easy','random','hard'])