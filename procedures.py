import os
from os.path import join
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from datetime import datetime

from data.datasets import ReturnMapDataset, ImplicitUDataset, GeneratingFunctionDataset
from ml.models import ReLuModel
from ml.training import train_model
from dynamics import Orbit


class Training:
    def __init__(self, num_epochs=100, cs="Custom", type="ReturnMap", *args, **kwargs):
        self.num_epochs = num_epochs
        self.cs = cs
        self.type = type

        self.model = None
        self.training_loss = 0
        self.validation_loss = 0

        self.today = datetime.today().strftime("%Y-%m-%d")
        self.this_dir = os.path.dirname(__file__)

    def train(self):
        # relevant directories
        data_dir = join(self.this_dir, "data/raw", self.type, self.cs)
        model_dir = join(self.this_dir, "output/models",
                         self.type, self.cs, self.today)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if self.type == "ReturnMap":
            Dataset = ReturnMapDataset
            input_dim = 2
            output_dim = 2
        elif self.type == "ImplicitU":
            Dataset = ImplicitUDataset
            input_dim = 2
            output_dim = 1
        elif self.type == "GeneratingFunction":
            Dataset = GeneratingFunctionDataset
            input_dim = 2
            output_dim = 1
        else:
            return

        # datasets
        train_dataset = Dataset(join(data_dir, "train50k.npy"))
        validation_dataset = Dataset(join(data_dir, "validate10k.npy"))

        # model
        model = ReLuModel(input_dim=input_dim, output_dim=output_dim)

        # train
        model, training_loss, validation_loss = train_model(model,
                                                            train_dataset,
                                                            validation_dataset,
                                                            torch.nn.MSELoss(),
                                                            self.num_epochs,
                                                            dir=model_dir)

        self.model = model
        self.training_loss = training_loss
        self.validation_loss = validation_loss

    def plot_loss(self):
        graphics_dir = join(self.this_dir, "output/graphics",
                            self.type, self.cs, self.today)
        Path(graphics_dir).mkdir(parents=True, exist_ok=True)
        graphic_filename = join(graphics_dir, "loss.png")

        fig = plt.figure()
        plt.suptitle(self.type)
        plt.plot(self.training_loss, label="train", c="navy")
        plt.plot(self.validation_loss, label="validation", c="pink")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(loc="upper right")
        plt.savefig(graphic_filename)


def training_procedure(num_epochs=100, type="GeneratingFunction", cs="Custom"):
    training = Training(num_epochs=num_epochs, type=type, cs=cs)
    training.train()
    training.plot_loss()


def minimization_procedure(a, b, dir=None):
    filename = os.path.join(os.path.dirname(__file__), "output/models", dir, "model.pth")

    generating_function = ReLuModel(input_dim=2, output_dim=1)
    generating_function.load_state_dict(torch.load(filename)["model_state_dict"])

    # number of applications of return map
    n = 2

    # initialize an orbit
    orbit = Orbit(a=a, b=b, n=n)
    # orbit.set_u(u_map(orbit.pair_s()))

    optimizer = torch.optim.Adam([orbit.phi], lr=1e-3)

    print(generating_function(orbit.pair_phi()[0]))
    print(generating_function(orbit.pair_phi()))

    n_epochs = 0
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        action = torch.abs(torch.sum(generating_function(orbit.pair_phi())))
        action.backward()

        optimizer.step()

        print('Epoch: {}, Action: {:.4f}'.format(epoch+1, action))

        # test = [-orbit.u[0] + orbit.u[1], -orbit.u[1] + orbit.u[2]]
        # print(orbit.s)

    print(orbit.phi)
