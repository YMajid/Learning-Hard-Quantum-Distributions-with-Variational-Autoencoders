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

        if file_name == "easy_dataset":
            np.savez(output_location, easy_dset=data)
        if file_name == "hard_dataset":
            np.savez(output_location, hard_dset=data)
        if file_name == "random_dataset":
            np.savez(output_location, random_dset=data)

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
