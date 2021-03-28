# Imports
import sys
import json
import argparse


sys.path.append('src/utils')
sys.path.append("src/utils/gen_data")
import data_reader

def create_dataset():
    data_reader.create_dataset(4)

if __name__ == '__main__':

    print("Main File.")
