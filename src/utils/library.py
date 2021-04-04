import numpy as np


class Library:
    """
    Library Class.

    Used for writing and reading Numpy arrays.
    """

    def __init__(self, path='data/'):
        self.path = path

    def writer(self, data, file_name):
        """
        Args:
            - data: Dataset to be saved
            - file_name: Location where dataset will be saved to
        Returns:
        Raises:
        """
        output_location = self.path + file_name + '.npz'
        np.savez(output_location, data)
        return

    def reader(self, file_name):
        """
        Args:
            - file_name: Name of file to be read
        Returns:
            - Returns Numpy array of the data
        Raises:
        """
        input_location = self.path + file_name + '.npz'
        data = np.load(input_location, mmap_mode='r')
        return data
