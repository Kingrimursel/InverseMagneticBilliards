import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import Slider

from shapely.geometry import LineString

from setting import Table
from util import (get_initial_theta,
                  get_legend,
                  update_slider_range,
                  get_chi,
                  rotate_vector,
                  area_overlap,
                  intersection_parameters,
                  angle_between,
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
        chord = LineString([tuple(p-1000*v), tuple(p+1000*v)])
        return chord

    def step(self, N=1):
        """Take N steps of the orbit

        Args:
            N (int, optional): Number of steps to take. Defaults to 1.

        Returns:
            np.array: The coordinates corresponding to the orbit
        """

        coordinates = [self.p]
        centers = []

        n_step = 0
        while N > 0:
            if self.mode == "classic":
                # get time of collision with boundary

                chord = self.get_chord(self.p, self.v)

                p2 = self.table.get_collision(chord)

                coordinates.append(p2)

                # p0 = self.p
                # t = self.table.get_collision(chord)

                # get collision point
                # p2 = self.p + t*self.v

                # coordinates.append(p2)

                # increase step counter by one
                N -= 1

            elif self.mode == "inversemagnetic":
                # corresponds to a chord
                if n_step % 2 == 0:
                    # get time of collision, the parameter of the straight line
                    p0 = coordinates[-1]

                    # FIXME: i think it might be choosing the wrong solution here sometimes i think. To make sure,
                    # just choose the one further away from self.p?!
                    t = self.table.get_collision(p0, self.v)

                    # get collision point
                    p1 = p0 + t*self.v

                    coordinates.append(p1)

                    # do not increase the step counter because this is only a temporary step
                    N -= 0

                # corresponds to a magnetic arc
                else:
                    p0 = coordinates[-2]
                    p1 = coordinates[-1]

                    # get larmor center
                    center = self.get_larmor_center(p0, p1)

                    centers.append(center)

                    # get reenter point
                    p2 = self.table.get_reenter_point(center, self.mu, p1)
                    coordinates.append(p2)

                    # increase step counter by one
                    N -= 1

            n_step += 1

        # stack coordinates
        coordinates = np.stack(coordinates)

        # return coordinates in classical case and coordinates/centers in inverse magnetic case
        if self.mode == "classic":
            return coordinates
        elif self.mode == "inversemagnetic":
            return coordinates, centers
        else:
            return

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

        center = center1 if is_left_of(p1-p0, center1) else center2

        return center

    def plot(self, img_path=None):
        fig, ax = plt.subplots()

        ax.add_patch(self.table.get_patch(fill="white"))
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

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


class Action:
    def __init__(self, a, b, mu, mode="classic", *args, **kwargs):
        self.mode = mode
        self.a = a
        self.b = b
        self.mu = mu
        self.kwargs = kwargs

        self.table = Table(a=a, b=b)

    def __call__(self, phi0s, theta0):
        orbit = Orbit(self.a, self.b, self.mu,
                      frequency=(1, 1), mode=self.mode, **self.kwargs)

        Gs = []
        phi2s = []

        for phi0, theta0 in zip(phi0s, theta0):
            phi0 = 0.
            theta0 = np.pi/2
            orbit.update(phi0, theta0)

            if self.mode == "classic":
                coordinates = orbit.step(N=1)
                G = np.linalg.norm(coordinates[0] - coordinates[1], ord=2)
                phi2 = self.table.get_polar_angle(coordinates[1])
            elif self.mode == "inversemagnetic":
                coordinates, centers = orbit.step(N=1)

                # ellipse parameters corresponding to intersection points
                phi2 = self.table.get_polar_angle(coordinates[2])

                # Area inside the circular arc but outside of the billiard table
                S = np.pi*self.mu**2 - \
                    area_overlap(self.a, self.b, self.mu, centers[0])

                phii, phif = intersection_parameters(
                    self.a, self.b, self.mu, centers[0])

                # length of first chord
                l1 = np.linalg.norm(coordinates[1] - coordinates[0], ord=2)

                # length of circular arc outside of the billiard table
                length_gamma = np.abs(self.mu * (phif - phii))

                # the action action
                G = - l1 - length_gamma + S
                print(l1, length_gamma, S, G)

            Gs.append(G)
            phi2s.append(phi2)

        return phi0s, phi2s, Gs


def periodic_orbits(a, b, mu):
    from trash import Trajectory
    # orbit properties
    m = 5
    n = 8
    N = 2*n-1

    # initial conditions
    s0 = 0
    theta0 = get_initial_theta(mu=mu, m=m, n=n).root

    # Plot dynamics
    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.axis("off")

    trajectory = Trajectory(s0, theta0, mu, a=a, b=b, mode="InverseMagnetic")
    trajectory.plot(ax, N=N, legend=get_legend(a, b, m, n))

    m_n_max = 16

    # Slider
    m_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    m_slider = Slider(
        ax=m_slider_ax,
        valstep=np.arange(1, m_n_max),
        label='m',
        valmin=1,
        valmax=m_n_max,
        valinit=m,
    )

    n_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    n_slider = Slider(
        ax=n_slider_ax,
        valstep=np.arange(1, m_n_max),
        label='n',
        valmin=1,
        valmax=m_n_max,
        valinit=n,
    )

    update_slider_range(m, n, m_slider, n_slider, m_n_max)

    def update(val):
        # orbit properties
        m = m_slider.val
        n = n_slider.val
        N = 2*n-1

        # initial conditions
        s0 = 0
        theta0 = get_initial_theta(mu=mu, m=m, n=n).root

        # Plot dynamics
        ax.clear()
        ax.axis("equal")
        ax.axis("off")

        traj = Trajectory(s0, theta0, mu, a=a, b=b)
        traj.plot(ax, N=N, legend=get_legend(a, b, m, n))

        # update slider
        update_slider_range(m, n, m_slider, n_slider, m_n_max)

        fig.canvas.draw()

    m_slider.on_changed(update)
    n_slider.on_changed(update)

    plt.show()
    plt.savefig("figure.png")
