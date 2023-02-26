from itertools import count, islice

import torch
from torch.utils import data

from glow.nn import get_loader


class Dataset(data.Dataset):
    def __getitem__(self, index):
        return index,

    def __len__(self):
        return 1


class Sampler(data.Sampler):
    def __init__(self, n):
        self.n = n
        self.count = count()

    def __iter__(self):
        return iter(islice(self.count, self.n))

    def __len__(self):
        return self.n


def test_loader():
    loader = get_loader(Dataset()).shuffle(Sampler(5)).batch(1)
    assert len(loader) == 5
    assert torch.as_tensor([*loader]).tolist() == [[0], [1], [2], [3], [4]]
    assert len(loader) == 5
    assert torch.as_tensor([*loader]).tolist() == [[5], [6], [7], [8], [9]]
