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

    # create dataset if not available locally (only takes a minute or three)
    if not os.path.exists('data/easy_dataset.npz'):
        create_dataset()

    # train_loaders, test_loaders = get_data()
    Model(parameters)
