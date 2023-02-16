import os
import torch

from data.generate import generate_dataset
from data.datasets import load_dataset
from ml.models import ReLuModel
from ml.training import train_model


if __name__ == "__main__":
    # table properties
    a = 2
    b = 1

    # magnetic properties
    mu = 1/5

    dataset = load_dataset(os.path.join(os.path.dirname(
        __file__), "data/raw/train50k.npy"))

    model = ReLuModel()

    train_model(model, dataset, torch.nn.MSELoss(), 100)
