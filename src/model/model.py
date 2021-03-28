import torch


class Model:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
