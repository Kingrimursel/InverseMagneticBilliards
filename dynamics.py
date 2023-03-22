import os
import torch
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection

from shapely.geometry import LineString, Point, LinearRing
from shapely.ops import nearest_points

from setting import Table
from util import (rotate_vector,
                  pair,
                  get_tangent,
                  is_left_of)

from conf import RES_LCIRC


class Orbit:
    def __init__(self, a, b, mu, frequency=(), mode="classic", init="random", cs="Birkhoff", helicity="pos", *args, **kargs):
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
        self.helicity = helicity

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
            self.phi = 2*torch.pi*torch.rand(self.n, requires_grad=True)
            # self.phi = torch.tensor(np.random.uniform(
            #    low=offset, high=2*np.pi-offset, size=self.n).astype("float32"), requires_grad=True)
        elif init == "uniform":
            indices = 1 + \
                torch.remainder(((self.m)*torch.arange(self.n)), self.n)

            indices[indices == 0] = self.n

            self.phi = indices/self.n*2*torch.pi
            if self.helicity == "neg":
                self.phi = torch.flip(self.phi, dims=(0,))

            offset_phi = 2*torch.pi*torch.rand(1).repeat(self.n)
            # self.phi += offset_phi

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
        self.u = self.points(x=torch.roll(self.phi, -1)) - \
            self.points(x=self.phi)

        return self.u

    def points(self, x=None):
        if x is None:
            return self.table.boundary(self.phi)
        else:
            return self.table.boundary(x)

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

    def get_exit_points(self):
        """Get the exit points and the corresponding Larmor centers of an orbit given its reenter points

        Returns:
            np.array, np.array: The exit points and the centers
        """

        # get reenter points and append the first one to the end because of periodicity
        p2s = self.table.boundary(self.phi.detach()).numpy()
        p2s = np.vstack([p2s, p2s[0]])

        centers = []
        p1s = []

        # want to find the exit point for all exit points
        for i, p2 in enumerate(p2s):
            # initial point
            p0 = p2s[i-1]

            # skip first point
            if i == 0:
                continue

            distances_i = []
            centers_i = []
            p1s_i = []

            # all vertices of a circle with radius mu around the reenter point are candidates for the larmor center
            vertices = Point(p2).buffer(self.mu).exterior.coords

            # loop over all vertices, checking which corresponding larmor circle comes closest to the real one
            for vertex in vertices:
                vertex = np.array(vertex)

                # get larmor circle
                # TODO: implement base class get_other_point and then implement get_reenter_point using that
                # TODO: controll resolution
                lcirc = Point(vertex).buffer(self.mu, resolution=RES_LCIRC)

                # obtain exit point corresponding to the given reenter point and larmor circle
                p1 = self.table.get_reenter_point(self.mu, vertex, p2)

                # if trajectory does not twist properly, continue. Also there could be problems with polygonial approximation
                if p1[0] is None or not is_left_of(p1-p0, vertex-p0):
                    distances_i.append(np.inf)
                    centers_i.append([None, None])
                    p1s_i.append([None, None])
                    continue

                # get two closes vertices of larmor circle to exit point
                # lcirc_coords = np.array(lcirc.exterior.coords)[:-1]
                # distances_vertices = np.array(
                #    [Point(p1).distance(Point(p)) for p in lcirc_coords])
                # closest_vertices = lcirc_coords[np.argpartition(
                #    distances_vertices, 1)[0:2]]

                # approximate the larmor circle's tangent at the exit point
                # tangent = closest_vertices[0] - closest_vertices[1]
                # tangent = tangent/np.linalg.norm(tangent)

                # define the chord that goes through the exit point and is parallel to the approximated tangent
                # chord = LineString([tuple(closest_vertices[0] - factor*tangent),
                #                   tuple(closest_vertices[0] + factor*tangent)])

                _, chord = get_tangent(
                    p1, lcirc, factor=6*max(self.table.a, self.table.b))

                # calculate distance between chord and previous reenter point
                dist_chord = chord.distance(Point(p0))

                # only consider the vertex if the map twists correctly
                distances_i.append(dist_chord)
                centers_i.append(vertex)
                p1s_i.append(p1)

            # find the best approximation
            idx = np.argmin(distances_i)
            centers.append(centers_i[idx])
            p1s.append(p1s_i[idx])

        return np.array(p1s), np.array(centers)

    def plot(self, img_dir=None, show=True, with_tangents=False):
        fig, ax = plt.subplots(figsize=(5, 5))

        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0.)
        plt.margins(0.01, 0.01)

        ax.add_patch(self.table.get_patch(fill="white"))
        ax.set_xlim([- max(self.table.a, self.table.b) - 0.5,
                    max(self.table.a, self.table.b) + 0.5])
        ax.set_ylim([- max(self.table.a, self.table.b) - 0.5,
                    max(self.table.a, self.table.b) + 0.5])

        points0 = self.table.boundary(self.phi0.detach())
        points2 = self.table.boundary(self.phi.detach())
        ax.scatter(points2[:, 0], points2[:, 1], c="navy")
        ax.scatter(points0[:, 0], points0[:, 1], c="green")

        # annotate the points
        for i, (xi, yi) in enumerate(zip(points2[:, 0], points2[:, 1])):
            plt.annotate(f'{i + 1}', xy=(xi, yi), xytext=(1.2*xi, 1.2*yi))

        # plot tangents
        if with_tangents:
            for point in points2:
                phi = self.table.get_polar_angle(point)
                v = self.table.tangent(phi)
                chord = self.get_chord(point, v)
                ax.plot(*chord.xy, c="red")

        if self.mode == "classic":
            # plot the chords
            xx = np.hstack([points2[:, 0], points2[0, 0]])
            yy = np.hstack([points2[:, 1], points2[0, 1]])

            ax.plot(xx, yy, c="black")

        elif self.mode == "inversemagnetic":
            points1, centers = self.get_exit_points()

            ax.scatter(points1[:, 0], points1[:, 1], c="navy")
            ax.scatter(centers[:, 0], centers[:, 1], c="magenta")

            # plot larmor circles
            circles = PatchCollection([plt.Circle(tuple(center), self.mu, alpha=1,
                                                  edgecolor="navy", zorder=0) for center in centers])

            circles.set_facecolor([0, 0, 0, 0])
            circles.set_edgecolor([0, 0, 0.5, 1])
            circles.set_zorder(0)
            ax.add_collection(circles)

            ax.plot(np.vstack([points2[:, 0], points1[:, 0]]),
                    np.vstack([points2[:, 1], points1[:, 1]]), c="navy")

        if img_dir is not None:
            plt.savefig(os.path.join(img_dir, "orbit.png"))

        if show:
            plt.show()

        plt.close()

        return fig, ax


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
            p2 = self.table.get_other_collision(chord, p0)

            coordinates = np.stack([p0, p2])

            center = None
        elif self.mode == "inversemagnetic":
            # corresponds to a chord
            # get time of collision, the parameter of the straight line

            # chord corresponding to first part of trajectory
            chord = self.get_chord(p0, v0)

            # collision point
            p1 = self.table.get_other_collision(chord, p0)

            # get larmor center
            if p1[0] is not None:
                center = self.get_larmor_center(p0, p1)

                # get reenter point
                p2 = self.table.get_reenter_point(self.mu, center, p1)

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

    def plot(self, phi0, theta0, show=True):
        # apply the return
        coordinates, center = self.__call__(phi0, theta0)

        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        ax.plot(*self.table.polygon.exterior.xy)
        ax.scatter(coordinates[:, 0], coordinates[:, 1])

        if center is not None:
            circle = Point(center).buffer(self.mu)
            ax.plot(*circle.exterior.xy)
            ax.scatter(*center)

        for i, (xi, yi) in enumerate(zip(coordinates[:, 0], coordinates[:, 1])):
            plt.annotate(f'{i}', xy=(xi, yi), xytext=(1.0*xi, 1.0*yi))

        if show:
            plt.show()

        plt.close()

        return fig, ax

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
