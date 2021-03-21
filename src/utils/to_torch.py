import torch
from torch.utils.data import DataLoader, TensorDataset


def to_torch(data, batch_size=1):
    """
    Converts a Numpy array into a Torch DataLoader.

    Args:
        data: Numpy data array.
        batch_size: Batch size DataLoader will have.
    Returns:
        loader: DataLoader based on input data and batch size.
    """
    data = torch.from_numpy(data)
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size)

    return loader
