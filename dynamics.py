import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.offsetbox import AnchoredText
from matplotlib.widgets import Slider

from setting import Table
from util import get_initial_theta, get_legend, update_slider_range, get_chi, rotate_vector, angle_between


class Trajectory:
    def __init__(self, phi0, theta0, mu, a, b, mode="classic", cs="Birkhoff"):
        """Trajectory of a charged particle

        Args:
            phi0 (_type_): _description_
            theta0 (_type_): _description_
            mu (_type_): _description_
            a (_type_): _description_
            b (_type_): _description_
            mode (str, optional): If set to 'classic', normal billiards. Else, inverse magnetic billiards. Defaults to "classic".
        """
        # defining the billiards table
        self.table = Table(a=a, b=b)
        self.a = a
        self.b = b

        self.mode = mode
        self.cs = cs

        if self.mode != "classic":
            chi = get_chi(theta0, mu)

        # initial conditions
        s0 = self.table.get_arclength(phi0)

        t0 = self.table.tangent(phi0)
        v0 = rotate_vector(t0, theta0)

        self.phi0 = phi0
        self.theta = theta0
        self.s0 = s0

        # constants
        self.mu = mu
        if self.mode != "classic":
            self.theta = theta0
            self.chi = chi

        # initialize runtime variables
        self.s = s0
        self.u = -np.cos(theta0)
        self.phi = phi0
        self.v = v0
        self.p = self.table.boundary(phi0)
        self.n_step = 0

    def update(self, phi0, theta0):
        self.phi = phi0
        self.theta = theta0

        self.v = rotate_vector(self.table.tangent(phi0), theta0)
        self.p = self.table.boundary(phi0)
        self.s = self.table.get_arclength(phi0)
        self.u = -np.cos(theta0)

    def step(self, N=1):
        """
        Do n iterations of the return map.
        """

        vs = [self.v]
        ps = [self.p]

        coordinates = [[self.s, self.u]]

        while N > 0:
            if self.mode == "classic":
                # get time of collision with boundary
                t = self.table.get_collision(self.p, self.v)

                # get collision point
                self.p = self.p + t*self.v

                # ellipse parameter corresponding to collision point. Caution: cos is not injective!
                if self.p[1] >= 0:
                    phi1 = np.arccos(self.p[0]/self.table.a)
                else:
                    phi1 = 2*np.pi - np.arccos(self.p[0]/self.table.a)

                # caculate arclength
                s1 = self.table.get_arclength(phi1)

                # update runtime variables
                theta1 = angle_between(self.v, self.table.tangent(phi1))
                u1 = - np.cos(theta1)

                self.v = rotate_vector(self.table.tangent(phi1), theta1)
                self.s = s1
                self.u = u1
                self.phi = phi1

                if self.cs == "Birkhoff":
                    coordinates.append([s1, u1])
                elif self.cs == "Custom":
                    coordinates.append([phi1, theta1])
                else:
                    return
            else:
                # corresponds to a chord
                if self.n_step % 2 == 0:
                    # get time of collision, the parameter of the straight line
                    t = self.table.get_collision(self.p, self.v)

                    # get collision point
                    self.p = self.p + t*self.v

                # corresponds to a magnetic arc
                else:
                    # the direction of the l_2 chord
                    v_chord = rotate_vector(self.v, self.chi)
                    # intersection of l_2 with the boundary
                    t = self.table.get_collision(self.p, v_chord)
                    # move base point along l_2 chord
                    self.p = self.p + t*v_chord
                    # rotate the velocity by psi=2*chi
                    self.v = rotate_vector(self.v, 2*self.chi)

            self.n_step += 1
            N -= 1

            ps.append(self.p)
            vs.append(self.v)

        ps = np.stack(ps)
        vs = np.stack(vs)
        coordinates = np.stack(coordinates)

        if self.mode == "classic":
            return coordinates
        else:
            return ps, vs

    def plot(self, ax, N=10, legend=None):
        ps, vs = self.step(N=N)

        # plot billiards table
        ax.add_patch(self.table.get_patch(fill="white"))

        # plot exit and reentry points
        ax.scatter(ps[:, 0], ps[:, 1], c="purple", zorder=20)

        # plot the larmor centers and -circles
        larmor_centers = ps[1::2] + self.mu * \
            rotate_vector(vs[1::2].T, np.pi/2).T
        ax.scatter(larmor_centers[:, 0],
                   larmor_centers[:, 1], c="yellow", zorder=20)

        circles = PatchCollection([plt.Circle(tuple(larmor_center), self.mu, alpha=1,
                                              edgecolor="navy", zorder=0) for larmor_center in larmor_centers])

        circles.set_facecolor([0, 0, 0, 0])
        circles.set_edgecolor([0, 0, 0, 1])
        circles.set_zorder(0)
        ax.add_collection(circles)

        # plot the non-magnetic chords
        xx = np.vstack([ps[0::2][:, 0], ps[1::2][:, 0]])
        yy = np.vstack([ps[0::2][:, 1], ps[1::2][:, 1]])

        ax.plot(xx, yy, c="black")

        # plot trajectory properties
        if legend:
            text_box = AnchoredText(legend, frameon=True, loc=4, pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            ax.add_artist(text_box)


class Orbit:
    def __init__(self, a, b, n=2, mode="classic", init="random", *args, **kargs):
        self.mode = mode
        self.table = Table(a=a, b=b)

        self.s = []
        self.phi = []
        self.u = []


        if init == "random":
            offset = 1e-7
            self.phi = torch.tensor(np.random.uniform(
                low=offset, high=2*np.pi-offset, size=n).astype("float32"), requires_grad=True)
        elif init == "uniform":
            self.phi = torch.arange(n)/n*2*torch.pi
            self.phi.requires_grad_()
        else:
            return

        self.phi0 = self.phi.clone()

    def set_u(self, u):
        self.u = u

    def get_u(self):
        return self.u

    def get_s(self):
        return self.s

    def get_phi(self):
        return self.phi
    
    def points(self, x=None, dtype="NumPy"):
        if x is None:
            return self.table.boundary(self.phi, dtype=dtype)
        else:
            return self.table.boundary(x, dtype=dtype)

    def pairs(self, x, periodic=True):
        pairs = torch.cat((torch.unsqueeze(x, dim=1),
                           torch.unsqueeze(torch.roll(x, -1), dim=1)), 1)

        if not periodic:
            pairs = pairs[:-1]

        return pairs

    def pair_phi(self, periodic=True):
        return self.pairs(self.phi, periodic=periodic)

    def pair_s(self, periodic=True):
        return self.pairs(self.s, periodic=periodic)

    def plot(self):
        fig, ax = plt.subplots()

        ax.add_patch(self.table.get_patch(fill="white"))
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])

        points0 = self.table.boundary(self.phi0.detach().numpy()).T
        points = self.table.boundary(self.phi.detach().numpy()).T
        ax.scatter(points[:, 0], points[:, 1])
        ax.scatter(points0[:, 0], points0[:, 1])

        # plot the chords
        xx = np.hstack([points[0::2][:, 0], points[1::2][:, 0], points[0][0]])
        yy = np.hstack([points[0::2][:, 1], points[1::2][:, 1], points[0][1]])

        ax.plot(xx, yy, c="black")

        plt.show()


class Action:
    def __init__(self, a, b, mode="classic", *args, **kwargs):
        self.mode = mode
        self.table = Table(a=a, b=b)

    def __call__(self, phi0, phi1):
        if self.mode == "classic":
            G = np.linalg.norm(self.table.boundary(
                phi0) - self.table.boundary(phi1), axis=0)
        else:
            G = None

        return G


def periodic_orbits(a, b, mu):
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
