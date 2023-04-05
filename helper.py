import os
import numpy as np
import numdifftools as nd
from matplotlib import pyplot as plt

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.datasets import ReturnMapDataset, ImplicitUDataset, GeneratingFunctionDataset
from setting import Table
from physics import DiscreteAction, Action
from shapely.geometry import Point
from ml.training import train_model
from ml.models import ReLuModel
from util import (batch_hessian,
                  angle_between,
                  generate_readme,
                  get_todays_graphics_dir,
                  batch_jacobian,
                  mkdir,
                  get_tangent,
                  unit_vector,
                  pair, values_in_quantile)
from conf import device, MODELDIR, DATADIR, TODAY


class Training:
    def __init__(self,
                 num_epochs=100,
                 mode="classic",
                 train_dataset="train50k.npy",
                 batch_size=128,
                 subdir="",
                 save=True,
                 **kwargs):

        self.num_epochs = num_epochs
        self.subdir = subdir
        self.mode = mode
        self.train_dataset = train_dataset
        self.save = save
        self.batch_size = batch_size

        self.model = None
        self.training_loss = 0.
        self.validation_loss = 0.

        self.data_dir = os.path.join(DATADIR, self.mode, self.subdir)

        if self.save:
            self.model_dir = os.path.join(MODELDIR, self.mode, subdir, TODAY)
            mkdir(self.model_dir)
        else:
            self.model_dir = None

    def train(self):

        # datasets
        train_data_dir = os.path.join(self.data_dir, self.train_dataset)
        print(f"Loading Training Dataset {train_data_dir}")
        train_dataset = GeneratingFunctionDataset(train_data_dir)
        validation_dataset = GeneratingFunctionDataset(
            os.path.join(self.data_dir, "validate10k.npy"))

        # model
        model = ReLuModel(input_dim=2, output_dim=1)

        # train
        model, training_loss, validation_loss = train_model(model,
                                                            train_dataset,
                                                            validation_dataset,
                                                            torch.nn.MSELoss(),
                                                            self.num_epochs,
                                                            dir=self.model_dir,
                                                            batch_size=self.batch_size,
                                                            device=device)

        self.model = model
        self.training_loss = training_loss
        self.validation_loss = validation_loss

    def plot_loss(self):
        img_dir = get_todays_graphics_dir(self.mode, self.subdir)
        graphic_filename = os.path.join(img_dir, "training_loss.png")

        fig = plt.figure()
        plt.plot(self.training_loss, label="train", c="navy")
        plt.plot(self.validation_loss, label="validation", c="pink")
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        # plt.xscale("log")
        plt.legend(loc="upper right")
        plt.savefig(graphic_filename)

        plt.show()

        plt.close()

    def generate_readme(self, a, b, k, mu, num_epochs, batch_size):
        generate_readme(
            self.model_dir, f"a={a},\nb={b},\nmu={mu}\nk={k}\nnum_epochs={num_epochs}\nbatch_size={batch_size}")


class Minimizer:
    def __init__(self, a, b, k, mu, orbit, action_fn, mode="classic", n_epochs=50, frequency=(), **kwargs):
        self.a = a
        self.b = b
        self.k = k
        self.table = Table(a=a, b=b, k=k)
        self.orbit = orbit
        self.n_epochs = n_epochs
        self.optimizer = torch.optim.Adam([orbit.phi], lr=1e-3)
        # self.optimizer = torch.optim.SGD([orbit.phi], lr=1e-2)
        self.action_fn = action_fn
        self.frequency = frequency
        self.m, self.n = frequency

        self.discrete_action = DiscreteAction(action_fn, self.table, **kwargs)
        self.action = Action(a, b, k, mu, mode=mode)

        self.grad_losses = []
        self.m_losses = []

    def minimize(self, exact_deriv=True):
        grad_losses = []

        grad_loss = torch.zeros(1)

        # if we do not calculate the exact derivative, detach
        # if not exact_deriv:
        #    self.orbit.phi.detach_()

        for epoch in (pb := tqdm(range(self.n_epochs))):
            self.optimizer.zero_grad()

            pb.set_postfix({"Loss": grad_loss.item()})

            # calculate the gradient loss
            # grad_loss = self.discrete_action.grad_norm(self.orbit.phi)
            # grad_loss = torch.abs(batch_jacobian(self.discrete_action, self.orbit.phi).sum())

            # do a gradient descent step
            if exact_deriv:
                grad_loss = self.discrete_action.grad_norm_summed(
                    self.orbit.phi)
                grad_loss.backward()
            else:
                with torch.no_grad():
                    grad_loss = self.discrete_action.grad_norm_summed(
                        self.orbit.phi.detach())
                    # grad_grad_loss = torch.from_numpy(nd.Jacobian(self.discrete_action.grad_norm_summed)(self.orbit.phi.detach())).squeeze().float()
                    # grad_grad_loss = jacobian_approx(self.discrete_action.grad_norm_summed, self.orbit.phi.detach())
                    grad_grad_loss = batch_jacobian(
                        self.discrete_action.grad_norm_summed, self.orbit.phi.detach(), approx=True)

                    self.orbit.phi.grad = grad_grad_loss

            self.optimizer.step()

            # since we have only learned the model in the interval [0, 2pi], we move the point there
            with torch.no_grad():
                self.orbit.phi.remainder_(2*torch.pi)

            # save losses
            grad_losses.append(grad_loss.item())

        self.grad_losses = torch.tensor(grad_losses)

    def plot(self, img_dir=None, show=True):
        print("DIAGNOSTIC minimizer...")

        fig = plt.figure()
        fig.suptitle("Minimization Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.plot(self.grad_losses)
        plt.yscale("log")
        # if len(self.m_losses) > 0:
        #    plt.plot(self.m_losses)

        if img_dir is not None:
            plt.savefig(os.path.join(img_dir, "minimization_loss.png"))

        if show:
            plt.show()

        plt.close()


class Diagnostics:
    def __init__(self, a, b, k, mu, orbit=None, mode="classic", subdir="", *args, **kwargs):
        self.orbit = orbit

        self.mode = mode
        self.subdir = subdir
        self.a = a
        self.b = b
        self.k = k
        self.mu = mu

        self.table = Table(a=self.a, b=self.b, k=self.k)

        self.data_dir = os.path.join(DATADIR, self.mode, self.subdir)

    def reflection_angle(self, unit="rad"):
        us = self.orbit.get_u()
        tangents = self.orbit.table.tangent(self.orbit.phi).T
        # tangents = torch.roll(tangents, -1, dims=0)

        einfallswinkel = []
        ausfallswinkel = []

        for i, tangent in enumerate(tangents):
            # print(us[i-1])
            # print(tangent)
            # print(us[i])
            # print(angle_between(us[i-1], tangent)*180/np.pi)
            # print(angle_between(us[i], tangent)*180/np.pi)
            # print("\n")
            # fig, ax = self.orbit.plot(show=False)
            # ax.quiver(*self.orbit.points()[i].detach().T, 10*us[i, 0].detach(), 10*us[i, 1].detach(), scale=1)
            # ax.quiver(*self.orbit.points()[i].detach().T, 10*us[i-1, 0].detach(), 10*us[i-1, 1].detach(), scale=1)
            # ax.quiver(*self.orbit.points()[i].detach().T, 10*tangents[i, 0].detach(), 10*tangents[i, 1].detach(), scale=1)
            # plt.show()
            einfallswinkel.append(angle_between(us[i-1], tangent))
            ausfallswinkel.append(angle_between(us[i], tangent))

        einfallswinkel = torch.tensor(einfallswinkel)
        ausfallswinkel = torch.tensor(ausfallswinkel)

        if unit == "deg":
            einfallswinkel = einfallswinkel/2/np.pi*360
            ausfallswinkel = ausfallswinkel/2/np.pi*360

        return einfallswinkel, ausfallswinkel

    def physics(self, unit="deg", img_dir=None, show=True):
        print(f"DIAGNOSTIC physics...")
        fig = plt.figure()
        # fig.suptitle("Error in Reflection Law")
        plt.xlabel("Point")

        if self.mode == "classic":
            plt.ylabel("Error [%]")
            einfallswinkel, ausfallswinkel = self.reflection_angle(unit=unit)

            error = torch.abs(
                100*(einfallswinkel - ausfallswinkel)/torch.max(einfallswinkel, ausfallswinkel))

            x = np.arange(len(error)) + 1
            plt.plot(x, error.detach(), label="Angle Error")

            print(f"Angles of Incidence: {einfallswinkel.tolist()}")
            print(f"Angles of Reflection: {ausfallswinkel.tolist()}")

        elif self.mode == "inversemagnetic":
            plt.ylabel("Error")
            p1s, centers = self.orbit.get_exit_points()
            p0s = self.orbit.points().detach().numpy()

            # p0s = np.roll(p0s, 1, axis=0)

            factor = 6*max(self.a, self.b)

            ex_errors = []
            re_errors = []

            for i in range(len(centers)):
                center = centers[i-1]
                circle = Point(center).buffer(self.mu)
                emp_ex_tan, _ = get_tangent(p1s[i-1], circle, factor=factor)
                emp_re_tan, _ = get_tangent(p0s[i], circle, factor=factor)
                re_ex_tan = unit_vector(p0s[i-1] - p1s[i-1])
                re_re_tan = unit_vector(p1s[i] - p0s[i])

                ex_error = 1 - abs(np.dot(emp_ex_tan, re_ex_tan))
                re_error = 1 - abs(np.dot(emp_re_tan, re_re_tan))

                ex_errors.append(ex_error)
                re_errors.append(re_error)

                # fig, ax = self.orbit.plot(show=False)
                # ax.quiver(*p0s[i].T, 10*emp_re_tan[0], 10*emp_re_tan[1], scale=1)
                # ax.quiver(*p1s[i-1].T, 10*emp_ex_tan[0], 10*emp_ex_tan[1], scale=1)
                # ax.quiver(*p1s[i-1].T, 10*re_ex_tan[0],
                #          10*re_ex_tan[1], scale=1)
                # ax.plot(*emp_ex_chord.xy)
                # plt.show()

            x = np.arange(len(ex_errors)) + 1
            plt.plot(x, ex_errors, label=r"$\Delta_1$")
            plt.plot(x, re_errors, label=r"$\Delta_2$")
            plt.ylim(0, 0.1)
            plt.gca().spines[['right', 'top']].set_visible(False)

        plt.xticks(x, x)
        plt.legend(loc="best")

        if img_dir is not None:
            plt.savefig(os.path.join(img_dir, "physics_error.png"))

        if show:
            plt.show()

        plt.close()

    def landscape(self, fn, n=150, repeat=False, dim=2, img_dir=None, show=True, plot_points=True, filename="landscape.png"):
        print("DIAGNOSTIC landscape...")
        phi0s = torch.linspace(0, 2*torch.pi, n)
        phi2s = torch.linspace(0, 2*torch.pi, n)

        grid_x, grid_y = torch.meshgrid(phi0s, phi2s, indexing="ij")
        coordinates = torch.vstack(
            [torch.ravel(grid_x), torch.ravel(grid_y)]).T

        G = torch.squeeze(fn(coordinates))

        if repeat:
            coordinates = coordinates.repeat(3, 1)
            coordinates[n**2:2*n**2, 0] += 2*torch.pi
            coordinates[2*n**2:, 1] += 2*torch.pi
            G = G.repeat(3)

        if dim == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection="3d")
            ax.scatter(coordinates[:, 0].detach(),
                       coordinates[:, 1].detach(),
                       G.detach(), c=G.detach())
        else:
            fig, ax = plt.subplots()

            ax.set_aspect("equal")
            ax.set_xticks([0, np.pi, 2*np.pi], ["0", "$\pi$", "$2\pi$"])
            ax.set_yticks([0, np.pi, 2*np.pi], ["0", "$\pi$", "$2\pi$"])
            ax.set_xlabel(r"$\varphi_0$")
            ax.set_ylabel(r"$\varphi_2$")

            scatter = ax.scatter(coordinates[:, 0].detach(),
                                 coordinates[:, 1].detach(), c=G.detach())

            orbit_pairs = pair(self.orbit.phi.detach())

            if plot_points:
                ax.scatter(orbit_pairs[:, 0],
                           orbit_pairs[:, 1], c="red", marker="x")

                for i, (xi, yi) in enumerate(pair(self.orbit.phi.detach())):
                    plt.annotate(f'{i + 1}', xy=(xi, yi),
                                 xytext=(1.1*xi, 1.1*yi), c="red")

            fig.colorbar(scatter, ax=ax)

        if img_dir is not None:
            plt.savefig(os.path.join(img_dir, filename))

        if show:
            plt.show()

        plt.close()

    def error(self, model, show=True, img_dir=None):
        print("DIAGNOSTIC error...")
        train_data_dir = os.path.join(self.data_dir, "validate10k.npy")
        train_dataset = GeneratingFunctionDataset(train_data_dir)
        train_loader = DataLoader(
            train_dataset, batch_size=1024, shuffle=True, pin_memory=True)

        all_targets = []
        all_outputs = []
        all_inputs = []

        for inputs, targets in train_loader:
            output = model(inputs)
            all_targets.append(targets)
            all_outputs.append(output)
            all_inputs.append(inputs)

        all_targets = torch.cat(all_targets).squeeze().detach()
        all_outputs = torch.cat(all_outputs).squeeze().detach()
        all_inputs = torch.cat(all_inputs).squeeze().detach()

        error = torch.abs((all_targets - all_outputs)/all_targets)

        idx = values_in_quantile(error, 0.99)

        # scatter plot of error
        fig, ax = plt.subplots()
        ax.set_aspect("equal")
        ax.set_xticks([0, np.pi, 2*np.pi], ["0", "$\pi$", "$2\pi$"])
        ax.set_yticks([0, np.pi, 2*np.pi], ["0", "$\pi$", "$2\pi$"])
        ax.set_xlabel(r"$\varphi_0$")
        ax.set_ylabel(r"$\varphi_2$")

        scatter = ax.scatter(all_inputs[:, 0][idx],
                             all_inputs[:, 1][idx], c=error[idx])

        cbar = plt.colorbar(scatter, ax=ax)
        # cbar.ax.yaxis.set_ticklabels([f'{np.exp(v)}' for v in cbar.ax.yaxis.get_ticklocs()])
        # cbar.ax.yaxis.set_ticks(np.arange(0, np.log(error.max()) + 1, 1))
        # cbar.ax.yaxis.set_ticklabels([f'{int(np.exp(v)):,}' for v in cbar.ax.yaxis.get_ticklocs()])

        if img_dir is not None:
            plt.savefig(os.path.join(img_dir, "error.png"))

        if show:
            plt.show()

        # histogram plot of error
        fig, ax = plt.subplots()

        ax.hist(error.detach()[idx], color="navy",
                alpha=.5, edgecolor="navy", linewidth=.5)
        ax.set_xlabel(r"$|\frac{ G - \hat{G}}{\hat{G}}|$")
        ax.set_ylabel("# points")

        if img_dir is not None:
            plt.savefig(os.path.join(img_dir, "error_hist.png"))

        if show:
            plt.show()

    def frequency(self):
        print("DIAGNOSTIC frequency...")
        phi_centered = self.orbit.phi - self.orbit.phi[0]

        diff = phi_centered - torch.roll(phi_centered, -1)
        m = torch.sum(diff > 0).item()
        n = len(self.orbit)

        # m_approx = torch.sum(torch.nn.Sigmoid()(360*diff)).item()

        return (m, n)
