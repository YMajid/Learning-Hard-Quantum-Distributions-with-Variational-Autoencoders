# Imports
import sys
import json
import argparse
import os

sys.path.append('src/model')
sys.path.append('src/utils')
sys.path.append("src/utils/gen_data")

from model import Model
from create_dataset import create_dataset

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Learning Hard Quantum States Using a Variational AutoEncoder')
    parser.add_argument('-v', type=int, default=0, metavar='N', help='Verbosity (0 = all information, else = nothing).')
    parser.add_argument('-n', type=int, default=18, metavar='N', help='Number of qubits.')
    parser.add_argument('--result', type=str, default='result/', metavar='result/', help='Directory for output files.')
    parser.add_argument('--pretrained', type=bool, default=False, metavar='False', help='Load pretrained model.')
    parser.add_argument('--param', type=str, default='param/parameters.json', metavar='param/param.json',
                        help='Parameter file path.')
    args = parser.parse_args()

    # Read parameter JSON file, convert it into a Python dictionary
    with open(args.param) as f:
        parameters = json.loads(f.read())
        f.close()

    # Create dataset if not available locally (only takes a minute or three)
    if not args.pretrained or not os.path.exists('data/easy_dataset.npz'):
        print("Creating dataset, please wait one moment.")
        create_dataset(n_qubits=args.n)
    else:
        print("Dataset found.")

    # Load and plot fidelities or training
    if args.pretrained:
        for state in ['easy', 'random', 'hard']:
            fs = []
            for i in range(1, 6):
                m = Model(parameters, verbosity = args.v, state=state, n_qubits=args.n, n_layers=i, load=f"results/saved_model_{state}_L{i}")
                fs.append(m.fidelity)
            m.plot_fidelities(fs, state=state)
    else:
        for state in ['easy', 'random', 'hard']:
            for i in range(1, 6):
                m = Model(parameters, verbosity = args.v, state=state, n_qubits=args.n, n_layers=i)
