import torch
import numpy as np
from library import Library
from torch.utils.data import SubsetRandomSampler, DataLoader, TensorDataset


def get_data(batch_size=100, file_path='data/l4n4/'):
    """
    Args:
        - batch_size: Size of batches
        - file_path: Path of file location
    Returns:
        - train_loaders: Array of Torch DataLoaders representing quantum states for training
        - test_loaders: Array of Torch DataLoaders representing quantum states for testing
    Raises:
    """
    train_loaders, test_loaders = __to_torch(batch_size, file_path)

    return train_loaders, test_loaders


def __get_raw_data(file_path):
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

    raw_easy = library.reader('easy_dataset')['easy_dset']
    raw_hard = library.reader('hard_dataset')['hard_dset']
    raw_random = library.reader('random_dataset')['rand_dset']

    return raw_easy, raw_hard, raw_random


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


def __to_torch(batch_size, file_path):
    """
    Args:
        - batch_size: Size of batches
        - file_path: Path of file location
    Returns:
        - train_loaders: Array of Torch DataLoaders representing quantum states for training
        - test_loaders: Array of Torch DataLoaders representing quantum states for testin
    Raises:
    """
    raw_easy, raw_hard, raw_random = __get_raw_data(file_path)

    easy_tensor = torch.from_numpy(raw_easy)
    hard_tensor = torch.from_numpy(raw_hard)
    random_tensor = torch.from_numpy(raw_random)

    easy_dataset = TensorDataset(easy_tensor)
    hard_dataset = TensorDataset(hard_tensor)
    random_dataset = TensorDataset(random_tensor)

    train_loaders, test_loaders = [], []
    datasets = [easy_dataset, hard_dataset, random_dataset]

    for dataset in datasets:
        train_sampler, test_sampler = __get_samplers(dataset, 0.3)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders
