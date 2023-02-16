import torch
import numpy as np
from torch.utils.data import Dataset


class BirkhoffDataset(Dataset):
    def __init__(self, dataset):
        self.data = dataset[:, 0, :]
        self.targets = dataset[:, 1, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


def load_dataset(file):
    return BirkhoffDataset(torch.from_numpy(np.load(file)).float())
