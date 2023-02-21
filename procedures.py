import os
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime

from data.datasets import load_dataset
from ml.models import ReLuModel
from ml.training import train_model


def training_procedure(num_epochs=100):
    # datasets
    train_dataset = load_dataset(os.path.join(os.path.dirname(
        __file__), "data/raw/train50k.npy"))
    validation_dataset = load_dataset(os.path.join(
        os.path.dirname(__file__), "data/raw/validate10k.npy"))

    # model
    model = ReLuModel()

    # filename for model
    model_filename = os.path.join(os.path.dirname(
        __file__), "output/checkpoints/models", datetime.today().strftime('%Y-%m-%d'))
    Path(model_filename).mkdir(parents=True, exist_ok=True)
    model_filename = os.path.join(model_filename, "model.pth")

    graphic_filename = os.path.join(os.path.dirname(
        __file__), "output/graphics", datetime.today().strftime('%Y-%m-%d'))
    Path(graphic_filename).mkdir(parents=True, exist_ok=True)
    graphic_filename = os.path.join(graphic_filename, "losses.png")

    _, train_losses, validation_losses = train_model(model,
                                                     train_dataset,
                                                     validation_dataset,
                                                     torch.nn.MSELoss(),
                                                     num_epochs,
                                                     filename=model_filename)

    # plot loss curves
    fig = plt.figure()
    plt.plot(train_losses, label="train", c="navy")
    plt.plot(validation_losses, label="validation", c="pink")
    plt.legend(loc="upper right")
    plt.savefig(graphic_filename)


def minimization_procedure(filename):
    return_map = ReLuModel()
    return_map.load_state_dict(torch.load(filename)["model_state_dict"])

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
