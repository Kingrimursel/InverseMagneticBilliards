import os
from os.path import join
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime

from data.datasets import ReturnMapDataset, ImplicitUDataset
from ml.models import ReLuModel
from ml.training import train_model
from dynamics import Orbit


def training_procedure(num_epochs=100, reldir=""):
    # relevant directories
    today = datetime.today().strftime("%Y-%m-%d")
    this_dir = os.path.dirname(__file__)
    data_dir = join(this_dir, "data/raw")
    model_dir = join(this_dir, "output/models", reldir, today)
    graphics_dir = join(this_dir, "output/graphics", reldir, today)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(graphics_dir).mkdir(parents=True, exist_ok=True)

    # datasets
    rm_train_dataset = ReturnMapDataset(join(data_dir, "train50k.npy"))
    rm_validation_dataset = ReturnMapDataset(join(data_dir, "validate10k.npy"))

    u_train_dataset = ImplicitUDataset(join(data_dir, "train50k.npy"))
    u_validation_dataset = ImplicitUDataset(join(data_dir, "validate10k.npy"))

    # model
    rm_model = ReLuModel(input_dim=2, output_dim=2)
    u_model = ReLuModel(input_dim=2, output_dim=1)

    # filename for model
    rm_model_dir = os.path.join(model_dir, "ReturnMap")
    u_model_dir = os.path.join(model_dir, "ImplicitU")
    Path(rm_model_dir).mkdir(parents=True, exist_ok=True)
    Path(u_model_dir).mkdir(parents=True, exist_ok=True)

    rm_graphic_filename = os.path.join(graphics_dir, "rm_losses.png")
    u_graphic_filename = os.path.join(graphics_dir, "u_losses.png")

    # train both models
    _, rm_train_losses, rm_validation_losses = train_model(rm_model,
                                                           rm_train_dataset,
                                                           rm_validation_dataset,
                                                           torch.nn.MSELoss(),
                                                           num_epochs,
                                                           dir=rm_model_dir)

    _, u_train_losses, u_validation_losses = train_model(u_model,
                                                         u_train_dataset,
                                                         u_validation_dataset,
                                                         torch.nn.MSELoss(),
                                                         num_epochs,
                                                         dir=u_model_dir)
    # plot loss curves
    fig = plt.figure()
    plt.suptitle("Return Map")
    plt.plot(rm_train_losses, label="train", c="navy")
    plt.plot(rm_validation_losses, label="validation", c="pink")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(loc="upper right")
    plt.savefig(rm_graphic_filename)

    fig = plt.figure()
    plt.suptitle("Implicit U")
    plt.plot(u_train_losses, label="train", c="navy")
    plt.plot(u_validation_losses, label="validation", c="pink")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(loc="upper right")
    plt.savefig(u_graphic_filename)


def minimization_procedure(a, b, rm_filename=None, u_filename=None):
    return_map = ReLuModel(input_dim=2, output_dim=2)
    return_map.load_state_dict(torch.load(rm_filename)["model_state_dict"])

    u_map = ReLuModel(input_dim=2, output_dim=1)
    u_map.load_state_dict(torch.load(u_filename)["model_state_dict"])

    # #applications of return map
    n = 2

    # initialize an orbit
    orbit = Orbit(a=a, b=b)
    orbit.init_s(n)
    orbit.set_u(u_map(orbit.pair_s()))

    optimizer = torch.optim.Adam(orbit.parameters(), lr=1e-3)

    n_epochs = 1

    for epoch in range(n_epochs):

        print(orbit.u)
        # test = [-orbit.u[0] + orbit.u[1], -orbit.u[1] + orbit.u[2]]
        # print(orbit.s)
        print(orbit.u)
