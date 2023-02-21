import torch
import numpy as np
from torch.utils.data import Dataset


class ReturnMapDataset(Dataset):
    def __init__(self, file):
        dataset = torch.from_numpy(np.load(file)).float()
        self.data = dataset[:, 0, :]
        self.targets = dataset[:, 1, :]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class ImplicitUDataset(Dataset):
    def __init__(self, file):
        dataset = torch.from_numpy(np.load(file)).float()
        self.data = dataset[:, :, 0]
        self.targets = dataset[:, 0, 1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
