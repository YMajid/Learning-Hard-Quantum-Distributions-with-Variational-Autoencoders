# Imports
import sys
import json
import argparse

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

    # create_dataset()

    # train_loaders, test_loaders = get_data()
    Model(parameters)
