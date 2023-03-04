import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from pathlib import Path
from datetime import datetime
from matplotlib import pyplot as plt

from data.datasets import ReturnMapDataset, ImplicitUDataset, GeneratingFunctionDataset
from physics import DiscreteAction
from dynamics import Orbit
from ml.training import train_model
from ml.models import ReLuModel
from util import batch_jacobian, batch_hessian, angle_between, sigmoid_scaled, generate_readme, mkdir
from conf import device, MODELDIR, DATADIR, TODAY, GRAPHICSDIR


class Training:
    def __init__(self,
                 num_epochs=100,
                 cs="custom",
                 type="ReturnMap",
                 mode="classic",
                 train_dataset="train50k.npy",
                 batch_size=128,
                 subdir="",
                 save=True,
                 alpha=1e-3,
                 *args,
                 **kwargs):

        self.num_epochs = num_epochs
        self.cs = cs
        self.subdir = subdir
        self.type = type
        self.mode = mode
        self.train_dataset = train_dataset
        self.save = save
        self.alpha = alpha
        self.batch_size = batch_size

        self.model = None
        self.training_loss = 0.
        self.hess_train_loss = 0.
        self.validation_loss = 0.

        self.data_dir = os.path.join(
            DATADIR, self.type, self.cs, self.mode, self.subdir)

        if self.save:
            self.model_dir = os.path.join(
                MODELDIR, self.type, self.cs, self.mode, subdir, TODAY)
            mkdir(self.model_dir)
        else:
            self.model_dir = None

    def train(self):
        if self.type == "ReturnMap":
            Dataset = ReturnMapDataset
            input_dim = 2
            output_dim = 2
        elif self.type == "ImplicitU":
            Dataset = ImplicitUDataset
            input_dim = 2
            output_dim = 1
        elif self.type == "generatingfunction":
            Dataset = GeneratingFunctionDataset
            input_dim = 2
            output_dim = 1
        else:
            return

        # datasets
        train_dataset = Dataset(os.path.join(
            self.data_dir, self.train_dataset))
        validation_dataset = Dataset(
            os.path.join(self.data_dir, "validate10k.npy"))

        # model
        model = ReLuModel(input_dim=input_dim, output_dim=output_dim)

        # train
        model, training_loss, validation_loss, hess_train_loss = train_model(model,
                                                                             train_dataset,
                                                                             validation_dataset,
                                                                             torch.nn.MSELoss(),
                                                                             self.num_epochs,
                                                                             dir=self.model_dir,
                                                                             alpha=self.alpha,
                                                                             batch_size=self.batch_size,
                                                                             device=device)

        self.model = model
        self.training_loss = training_loss
        self.validation_loss = validation_loss
        self.hess_train_loss = hess_train_loss

    def plot_loss(self):
        graphics_dir = os.path.join(
            GRAPHICSDIR, self.type, self.cs, self.subdir, TODAY)
        mkdir(graphics_dir)
        graphic_filename = os.path.join(graphics_dir, "loss.png")

        fig = plt.figure()
        plt.suptitle(self.type)
        plt.plot(self.training_loss, label="train", c="navy")
        plt.plot(self.validation_loss, label="validation", c="pink")
        plt.plot(self.hess_train_loss, label="hessian train", c="green")
        plt.yscale("log")
        # plt.xscale("log")
        plt.legend(loc="upper right")
        plt.savefig(graphic_filename)

        plt.show()

    def generate_readme(self, a, b, mu, num_epochs, batch_size):
        generate_readme(
            self.model_dir, f"a={a},\nb={b},\nmu={mu}\nnum_epochs={num_epochs}\nbatch_size={batch_size}")


class Minimizer:
    def __init__(self, orbit, action_fn, n_epochs=50, frequency=(), *args, **kwargs):
        self.orbit = orbit
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam([orbit.phi], lr=1e-3)
        self.action_fn = action_fn
        self.frequency = frequency
        self.m, self.n = frequency

        self.discrete_action = DiscreteAction(action_fn, orbit=orbit, **kwargs)

        self.grad_losses = []
        self.m_losses = []

    def minimize(self):
        grad_losses = []
        m_losses = []

        grad_loss = torch.zeros(1)

        for epoch in (pb := tqdm(range(self.n_epochs))):
            self.optimizer.zero_grad()

            pb.set_postfix({"Loss": grad_loss.item()})

            # calculate the gradient loss
            grad_loss = torch.linalg.norm(
                batch_jacobian(self.discrete_action, self.orbit.phi))
            # print(batch_jacobian(self.discrete_action, self.orbit.phi).shape)
            # grad_loss = batch_jacobian(self.discrete_action, self.orbit.phi).sum()

            # a loss that should enforce the correct frequency?!
            if len(self.frequency) != 0:
                phi_centered = self.orbit.phi - self.orbit.phi[0]
                diff = phi_centered - torch.roll(phi_centered, -1)
                m_approx = torch.sum(sigmoid_scaled(diff, alpha=10))
                m_loss = (m_approx - self.m)**2
            else:
                m_loss = torch.tensor([0], requires_grad=True)

            total_loss = grad_loss

            # do a gradient descent step
            total_loss.backward()
            self.optimizer.step()

            # since we have only learned the model in the interval [2, 2pi], we move the point there
            with torch.no_grad():
                self.orbit.phi.remainder_(2*np.pi)

            # save losses
            grad_losses.append(grad_loss.item())
            m_losses.append(m_loss)

            # log the result
            # print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, grad_loss))

        self.grad_losses = torch.tensor(grad_losses)
        self.m_losses = torch.tensor(m_losses)

    def plot(self):
        fig = plt.figure()
        fig.suptitle("Minimization Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.grad_losses)
        # if len(self.m_losses) > 0:
        #    plt.plot(self.m_losses)

        plt.show()


class Diagnostics:
    def __init__(self, orbit=None, cs="custom", type="ReturnMap", mode="classic", *args, **kwargs):
        self.orbit = orbit

        self.cs = cs
        self.type = type
        self.mode = mode

    def reflection_angle(self, unit="rad"):
        us = self.orbit.get_u()
        tangents = self.orbit.table.tangent(self.orbit.phi).T

        einfallswinkel = angle_between(us, tangents)
        ausfallswinkel = angle_between(torch.roll(tangents, -1, dims=0), us)

        if unit == "deg":
            einfallswinkel = einfallswinkel/2/np.pi*360
            ausfallswinkel = ausfallswinkel/2/np.pi*360

        return einfallswinkel, ausfallswinkel

    def reflection(self, unit="deg"):
        einfallswinkel, ausfallswinkel = self.reflection_angle(unit=unit)
        error = torch.abs(100*(einfallswinkel - ausfallswinkel)/einfallswinkel)

        fig = plt.figure()
        fig.suptitle("Error in Reflection Law")
        plt.xlabel("Point")
        plt.ylabel("Error [%]")
        # plt.errorbar(np.arange(len(error)), error.detach(), fmt=".")
        plt.plot(error.detach())
        # plt.yscale("log")

        plt.show()

    def frequency(self):
        phi_centered = self.orbit.phi - self.orbit.phi[0]

        diff = phi_centered - torch.roll(phi_centered, -1)
        m = torch.sum(diff > 0).item()
        n = len(self.orbit)

        # m_approx = torch.sum(torch.nn.Sigmoid()(360*diff)).item()

        return (m, n)

    def derivative(self, a, b, dir):
        """Analyze the networks gradients

        Args:
            a (int): length of first semi axis
            b (int): length of second semi axis
            dir (string): directory where trained model is stored
            ord (int, optional): order of derivative. 1 corresponds to the Jacobian, 2 to the Hessian. Defaults to 1.
        """
        model_dir = os.path.join(MODELDIR, dir)

        train_settings = torch.load(os.path.join(model_dir, "model.pth"))

        model = ReLuModel(input_dim=2, output_dim=1)

        data_dir = os.path.join(DATADIR, self.type, self.cs)

        validation_dataset = GeneratingFunctionDataset(
            os.path.join(data_dir, "validate10k.npy"))

        validation_loader = DataLoader(
            validation_dataset, batch_size=1024, shuffle=True, pin_memory=True)

        deriv_diff_losses = []
        deriv_emp_losses = []
        deriv_ex_losses = []

        for epoch in range(1, train_settings["epochs"] + 1):
            epoch_deriv_diff_loss = 0.0
            epoch_deriv_emp_loss = 0.0
            epoch_deriv_ex_loss = 0.0

            for inputs, targets in validation_loader:
                epoch_settings = torch.load(os.path.join(
                    model_dir, "epochs", str(epoch), "model.pth"))

                model.load_state_dict(epoch_settings["model_state_dict"])

                # calculate empirical jacobian
                deriv_emp = torch.squeeze(batch_hessian(model.model, inputs))

                # calculate exact jacobian
                orbit = Orbit(a, b)
                orbit.set_phi(inputs)

                discrete_action = DiscreteAction(None, orbit, exact=True)
                deriv_ex = torch.squeeze(
                    batch_hessian(discrete_action, inputs))

                # compute jacobian losses
                new_deriv_diff_loss = (
                    deriv_emp - deriv_ex).pow(2).mean().item()

                new_deriv_ex_loss = deriv_ex.pow(2).mean().item()
                new_deriv_emp_loss = deriv_emp.pow(2).mean().item()

                # save epoch loss
                epoch_deriv_diff_loss += new_deriv_diff_loss
                epoch_deriv_emp_loss += new_deriv_emp_loss
                epoch_deriv_ex_loss += new_deriv_ex_loss

            deriv_diff_loss = epoch_deriv_diff_loss / len(validation_dataset)
            deriv_emp_loss = epoch_deriv_emp_loss / len(validation_dataset)
            deriv_ex_loss = epoch_deriv_ex_loss / len(validation_dataset)
            deriv_diff_losses.append(deriv_diff_loss)
            deriv_emp_losses.append(deriv_emp_loss)
            deriv_ex_losses.append(deriv_ex_loss)

            print("Epoch: {}. Diff Loss: {:.4f}. Emp Loss:{:.4f}. Ex Loss: {:.4f}".format(
                epoch, deriv_diff_loss, deriv_emp_loss, deriv_ex_loss))

        fig = plt.figure()
        fig.suptitle("Hessian Errors")
        plt.xlabel("Epoch")
        plt.ylabel("Error")
        plt.plot(deriv_diff_losses, label="difference")
        plt.plot(deriv_emp_losses, label="empirical")
        plt.plot(deriv_ex_losses, label="exact")
        plt.legend()

        plt.show()
