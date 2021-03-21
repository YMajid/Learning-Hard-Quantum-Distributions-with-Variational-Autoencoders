import numpy as np


class Library:
    """
    Library Class.

    Used for writing and reading Numpy arrays.
    """

    def __init__(self, path='data/'):
        self.path = path

    def writer(self, data, file_name):
        output_location = self.path + file_name + '.npy'
        np.save(output_location, data)
        return

    def reader(self, file_name):
        input_location = self.path + file_name + '.npy'
        data = np.load(input_location)
        return data
