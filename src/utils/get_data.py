import torch
import numpy as np
from library import Library
from torch.utils.data import SubsetRandomSampler, DataLoader, TensorDataset


def get_data(batch_size=100, file_path='data/l2n4_bin/', state='hard'):
    """
    Args:
        - batch_size: Size of batches
        - file_path: Path of file location
    Returns:
        - train_loaders: Array of Torch DataLoaders representing quantum states for training
        - test_loaders: Array of Torch DataLoaders representing quantum states for testing
    Raises:
    """
    train_loaders, test_loaders = __to_torch(
        batch_size, file_path, state=state)

    return train_loaders, test_loaders


def __get_raw_data(file_path, state='hard'):
    """
    Args:
        - file_path: Path of file location
    Returns:
        - raw_easy: Numpy array of easy quantum states
        - raw_hard: Numpy array of hard quantum states
        - raw_random: Numpy array of random quantum states
    Raises:
    """
    library = Library(file_path)
    raw = library.reader(f'{state}_dataset')[f'{state}_dset']

    return raw


def __get_samplers(dataset, percent_test=0.3):
    """
    Args:
        - dataset: Dataset to be sampled
        - percent_test: Portion of dataset that will be used for testing
    Returns:
        - train_sampler: Indices of training data
        - test_sampler: Indices of testing data
    Raises:
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(percent_test * dataset_size))

    np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    return train_sampler, test_sampler


def __to_torch(batch_size, file_path, state='hard'):
    """
    Args:
        - batch_size: Size of batches
        - file_path: Path of file location
    Returns:
        - train_loaders: Array of Torch DataLoaders representing quantum states for training
        - test_loaders: Array of Torch DataLoaders representing quantum states for testin
    Raises:
    """
    raw = __get_raw_data(file_path, state=state)
    dataset = raw.astype(float)

    split = int(np.floor(0.9 * len(dataset)))
    train_loader = DataLoader(
        dataset[:split], batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        dataset[split:], batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
