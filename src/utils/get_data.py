import torch
from library import Library
from torch.utils.data import DataLoader, TensorDataset


def get_data(batch_size=100, file_path='data/l4n4/'):
    easy_loader, hard_loader, random_loader = __to_torch(batch_size, file_path)

    return easy_loader, hard_loader, random_loader


def __get_raw_data(file_path):
    library = Library(file_path)

    raw_easy = library.reader('easy_dataset')['easy_dset']
    raw_hard = library.reader('hard_dataset')['hard_dset']
    raw_random = library.reader('random_dataset')['rand_dset']

    return raw_easy, raw_hard, raw_random


def __to_torch(batch_size, file_path):
    raw_easy, raw_hard, raw_random = __get_raw_data(file_path)

    easy_tensor = torch.from_numpy(raw_easy)
    hard_tensor = torch.from_numpy(raw_hard)
    random_tensor = torch.from_numpy(raw_random)

    easy_dataset = TensorDataset(easy_tensor)
    hard_dataset = TensorDataset(hard_tensor)
    random_dataset = TensorDataset(random_tensor)

    easy_loader = DataLoader(easy_dataset, batch_size=batch_size)
    hard_loader = DataLoader(hard_dataset, batch_size=batch_size)
    random_loader = DataLoader(random_dataset, batch_size=batch_size)

    return easy_loader, hard_loader, random_loader
