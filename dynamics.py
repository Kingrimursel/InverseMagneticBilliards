import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import Slider

from shapely.geometry import LineString, Point

from setting import Table
from util import (rotate_vector,
                  pair,
                  is_left_of)


class Orbit:
    def __init__(self, a, b, mu, frequency=(), mode="classic", init="random", cs="Birkhoff", *args, **kargs):
        """An inverse magnetic orbit

        Args:
            a (int): first semi axis of the ellipse
            b (int): second semi axis of the ellipse
            frequency (tuple, optional): frequency of periodic orbit considered. Defaults to ().
            mode (str, optional): classic of birkhoff billiards. Defaults to "classic".
            init (str, optional): how to initialize the orbit. Defaults to "random".
            cs (str, optional): coordinate system, whether classic or birkhoff. Defaults to "Birkhoff".
        """

        self.mode = mode
        self.cs = cs
        self.mu = mu

        if len(frequency) > 0:
            self.m, self.n = frequency
        else:
            self.m, self.n = 0, 0

        self.table = Table(a=a, b=b)

        self.s = []
        self.phi = []
        self.u = []

        # initialize the orbit
        if init == "random":
            offset = 1e-7
            self.phi = torch.tensor(np.random.uniform(
                low=offset, high=2*np.pi-offset, size=self.n).astype("float32"), requires_grad=True)
        elif init == "uniform":
            indices = 1 + \
                torch.remainder(((self.m)*torch.arange(self.n)), self.n)

            indices[indices == 0] = self.n

            self.phi = indices/self.n*2*torch.pi
            self.phi.requires_grad_()
        else:
            return

        self.phi0 = self.phi.clone()

        self.p = self.points()

    def __len__(self):
        return len(self.phi)

    def set_u(self, u):
        self.u = u

    def set_phi(self, phi):
        self.phi = phi
        self.p = self.points()

    def get_u(self):
        phi = self.get_phi()
        self.u = self.points(x=phi) - self.points(x=torch.roll(phi, -1))

        return self.u

    def get_s(self):
        return self.s

    def get_phi(self):
        return self.phi

    def points(self, x=None):
        if x is None:
            return self.table.boundary(self.phi)
        else:
            return self.table.boundary(x)

    def pair_phi(self, periodic=True):
        return pair(self.phi, periodic=periodic)

    def pair_s(self, periodic=True):
        return pair(self.s, periodic=periodic)

    def update(self, phi0, theta0):
        self.phi = phi0
        self.theta = theta0

        self.v = rotate_vector(self.table.tangent(phi0), theta0)
        self.p = self.table.boundary(phi0)
        self.s = self.table.get_arclength(phi0)
        self.u = -np.cos(theta0)

    def get_chord(self, p, v):
        chord = LineString([tuple(p-10*v), tuple(p+10*v)])
        return chord

    def step(self, N=1):
        """Take N steps of the orbit

        Args:
            N (int, optional): Number of steps to take. Defaults to 1.

        Returns:
            np.array: The coordinates corresponding to the orbit
        """

        raise NotImplementedError

    def plot(self, img_path=None):
        print(self.mode)
        fig, ax = plt.subplots()

        ax.add_patch(self.table.get_patch(fill="white"))
        ax.set_xlim([- max(self.table.a, self.table.b) - 0.5,
                    max(self.table.a, self.table.b) + 0.5])
        ax.set_ylim([- max(self.table.a, self.table.b) - 0.5,
                    max(self.table.a, self.table.b) + 0.5])

        points0 = self.table.boundary(self.phi0.detach())
        points = self.table.boundary(self.phi.detach())
        ax.scatter(points[:, 0], points[:, 1])
        ax.scatter(points0[:, 0], points0[:, 1])

        # plot the chords
        xx = np.hstack([points[:, 0], points[0, 0]])
        yy = np.hstack([points[:, 1], points[0, 1]])

        ax.plot(xx, yy, c="black")
        ax.set_aspect("equal")

        for i, (xi, yi) in enumerate(zip(xx[:-1], yy[:-1])):
            plt.annotate(f'{i + 1}', xy=(xi, yi), xytext=(1.2*xi, 1.2*yi))

        if img_path is not None:
            plt.savefig(img_path)

        plt.show()


class ReturnMap:
    def __init__(self, a, b, mu, mode="classic", *args, **kwargs):
        self.a = a
        self.b = b
        self.mu = mu
        self.mode = mode

        self.table = Table(a=a, b=b)

    def get_chord(self, p, v):
        chord = LineString([tuple(p-10*v), tuple(p+10*v)])
        return chord

    def __call__(self, phi0, theta0):
        p0 = self.table.boundary(phi0)
        v0 = rotate_vector(self.table.tangent(phi0), theta0)

        if self.mode == "classic":
            # get linestring object corresponding to chord
            chord = self.get_chord(p0, v0)

            # collision point
            p2 = self.table.get_collision(chord)

            coordinates = np.stack([p0, p2])

            center = None
        elif self.mode == "inversemagnetic":
            # corresponds to a chord
            # get time of collision, the parameter of the straight line

            # chord corresponding to first part of trajectory
            chord = self.get_chord(p0, v0)

            # collision point
            p1 = self.table.get_collision(chord)

            # get larmor center
            # FIXME: this sometimes returns the wrong center!!
            if p1[0] is not None:
                center = self.get_larmor_center(p0, p1)

                # get reenter point
                p2 = self.table.get_reenter_point(
                    self.table, self.mu, center, p1)

                # TODO: vielleicht eine gemeinsame funktion um die intersection points zu bekommen, und die dann für get_reenter_point and get_parameters nutzen
                # TODO: function for larmor circle? Or even a class?

            else:
                center = [None, None]
                p2 = [None, None]

            # stack the points
            coordinates = np.stack([p0, p1, p2])

        else:
            coordinates, center = None, None

        return coordinates, center

    def plot(self, phi0, theta0):
        # apply the return
        coordinates, center = self.__call__(phi0, theta0)

        circle = Point(center).buffer(self.mu)

        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        ax.plot(*self.table.polygon.exterior.xy)

        if circle is not None:
            ax.plot(*circle.exterior.xy)
            ax.scatter(coordinates[:, 0], coordinates[:, 1])

        for i, (xi, yi) in enumerate(zip(coordinates[:, 0], coordinates[:, 1])):
            plt.annotate(f'{i}', xy=(xi, yi), xytext=(1.2*xi, 1.2*yi))

        plt.show()

    def get_larmor_centers(self, p0, p1):
        x0 = p0[0]
        y0 = p0[1]
        x1 = p1[0]
        y1 = p1[1]

        mu1x = (x0**2*x1 - 2*x0*x1**2 + x1**3 + x1*(y0 - y1)**2 - np.sqrt(self.mu**2*(y0 - y1)
                ** 2*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2)))/(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2)

        mu1y = (x0**2*y1*(y0 - y1) + x0*(2*x1*y1*(-y0 + y1) + np.sqrt(self.mu**2*(y0 - y1)**2*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2))) + x1**2*y1*(y0 - y1) -
                x1*np.sqrt(self.mu**2*(y0 - y1)**2*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2)) + y1*(y0 - y1)**3)/((y0 - y1)*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2))

        mu2x = (x0**2*x1 - 2*x0*x1**2 + x1**3 + x1*(y0 - y1)**2 + np.sqrt(self.mu**2*(y0 - y1)
                ** 2*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2)))/(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2)

        mu2y = (x0**2*y1*(y0 - y1) - x0*(2*x1*y1*(y0 - y1) + np.sqrt(self.mu**2*(y0 - y1)**2*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2))) + x1**2*y1*(y0 - y1) +
                x1*np.sqrt(self.mu**2*(y0 - y1)**2*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2)) + y1*(y0 - y1)**3)/((y0 - y1)*(x0**2 - 2*x0*x1 + x1**2 + (y0 - y1)**2))

        mu1 = [mu1x, mu1y]
        mu2 = [mu2x, mu2y]

        return mu1, mu2

    def get_larmor_center(self, p0, p1):
        center1, center2 = self.get_larmor_centers(p0, p1)

        center = center1 if is_left_of(p1-p0, center1-p0) else center2

        return center
