import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

from data.datasets import ReturnMapDataset, ImplicitUDataset, GeneratingFunctionDataset
from util import batch_jacobian, angle_between
from physics import DiscreteAction
from ml.training import train_model
from ml.models import ReLuModel


class Training:
    def __init__(self, num_epochs=100, cs="Custom", type="ReturnMap", train_dataset="train50k.npy", *args, **kwargs):
        self.num_epochs = num_epochs
        self.cs = cs
        self.type = type
        self.train_dataset = train_dataset

        self.model = None
        self.training_loss = 0
        self.validation_loss = 0

        self.today = datetime.today().strftime("%Y-%m-%d")
        self.this_dir = os.path.dirname(__file__)

    def train(self):
        # relevant directories
        data_dir = os.path.join(self.this_dir, "data/raw", self.type, self.cs)
        model_dir = os.path.join(self.this_dir, "output/models",
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
        train_dataset = Dataset(os.path.join(data_dir, self.train_dataset))
        validation_dataset = Dataset(os.path.join(data_dir, "validate10k.npy"))

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
        graphics_dir = os.path.join(self.this_dir, "output/graphics",
                                    self.type, self.cs, self.today)
        Path(graphics_dir).mkdir(parents=True, exist_ok=True)
        graphic_filename = os.path.join(graphics_dir, "loss.png")

        fig = plt.figure()
        plt.suptitle(self.type)
        plt.plot(self.training_loss, label="train", c="navy")
        plt.plot(self.validation_loss, label="validation", c="pink")
        plt.yscale("log")
        plt.xscale("log")
        plt.legend(loc="upper right")
        plt.savefig(graphic_filename)


class Minimizer:
    def __init__(self, orbit, action_fn, n_epochs=50, *args, **kwargs):
        self.orbit = orbit
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam([orbit.phi], lr=1e-2)
        self.action_fn = action_fn

        self.discrete_action = DiscreteAction(action_fn, orbit=orbit, **kwargs)

        self.grad_loss = []

    def minimize(self):
        grad_losses = []
        for epoch in range(self.n_epochs):
            self.optimizer.zero_grad()

            # calculate the gradient loss
            grad_loss = torch.linalg.norm(
                batch_jacobian(self.discrete_action, self.orbit.phi))

            total_loss = grad_loss

            # do a gradient descent step
            total_loss.backward()
            self.optimizer.step()

            # since we have only learned the model in the interval [2, 2pi], we move the point there
            with torch.no_grad():
                self.orbit.phi.remainder_(2*np.pi)

            # save losses
            grad_losses.append(grad_loss.item())

            # log the result
            print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, grad_loss))

        self.grad_losses = torch.tensor(grad_losses)

    def plot(self):
        fig = plt.figure()
        fig.suptitle("Minimization Loss")
        plt.plot(self.grad_losses)
        plt.xlabel("epoch")
        plt.ylabel("loss")

        plt.show()


class Diagnostics:
    def __init__(self, orbit, *args, **kwargs):
        self.orbit = orbit

    def reflection_angle(self, unit="rad"):
        us = self.orbit.get_u()
        tangents = self.orbit.table.tangent(self.orbit.phi).T

        einfallswinkel = angle_between(us, tangents)
        ausfallswinkel = angle_between(torch.roll(tangents, -1, dims=0), us)

        if unit == "deg":
            einfallswinkel = einfallswinkel/2/np.pi*360
            ausfallswinkel = ausfallswinkel/2/np.pi*360

        return einfallswinkel, ausfallswinkel

    def reflection_error(self, unit="deg"):
        einfallswinkel, ausfallswinkel = self.reflection_angle(unit=unit)
        error = torch.abs(100*(einfallswinkel - ausfallswinkel)/einfallswinkel)

        return error

    def frequency(self):
        phi_centered = self.orbit.phi - self.orbit.phi[0]

        diff = phi_centered - torch.roll(phi_centered, -1)
        m = torch.sum(diff > 0).item()

        n = len(self.orbit)

        return (m, n)

    def plot(self):
        error = self.reflection_error()

        fig = plt.figure()
        fig.suptitle("Error in Reflection Law")
        plt.xlabel("Point")
        plt.ylabel("Error [%]")
        # plt.errorbar(np.arange(len(error)), error.detach(), fmt=".")
        plt.plot(error.detach())
        # plt.yscale("log")

        plt.show()
