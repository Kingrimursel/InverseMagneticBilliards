import torch
import numpy as np

from setting import Table
from dynamics import ReturnMap
from util import pair, circ_int_params, area_overlap


class DiscreteAction:
    def __init__(self, action_fn, table, *args, **kwargs):
        self.action_fn = action_fn
        self.table = table

    def __call__(self, phis):
        action = torch.sum(self.action_fn(pair(phis)))
        return action


class Action:
    def __init__(self, a, b, k, mu, mode="classic", *args, **kwargs):
        self.mode = mode
        self.a = a
        self.b = b
        self.k = k
        self.mu = mu
        self.kwargs = kwargs

        self.table = Table(a=a, b=b, k=k)

        self.returnmap = ReturnMap(self.a,
                                   self.b,
                                   self.k,
                                   self.mu,
                                   frequency=(1, 1),
                                   mode=self.mode,
                                   **self.kwargs)
        
    def exact(self, phis):
        p0 = self.table.boundary(phis[:, 0])
        p2 = self.table.boundary(phis[:, 1])

        return - torch.linalg.norm(p0 - p2, dim=1)

    def __call__(self, phi, theta):
        coordinates, center = self.returnmap(phi, theta)

        if self.mode == "classic":
            if coordinates[1][0] is not None:
                G = - np.linalg.norm(coordinates[0] - coordinates[1], ord=2)
                phi2 = self.table.get_polar_angle(coordinates[1])
            else:
                G = None
                phi2 = None
        elif self.mode == "inversemagnetic":
            # could happen due to polynomial approximation
            if coordinates[2][0] is not None:
                # ellipse parameters corresponding to intersection points
                phi2 = self.table.get_polar_angle(coordinates[2])

                # Area inside the circular arc but outside of the billiard table
                S = np.pi*self.mu**2 - \
                    area_overlap(self.table, self.mu, center)

                assert S > 0

                phii, phif = circ_int_params(self.table, self.mu, center, phi2)

                phi_delta = phif - phii if phif > phii else 2 * \
                    torch.pi - (phii - phif)

                assert phi_delta > 0

                if phii is not None:
                    # length of first chord
                    # print(phii*180/torch.pi, phif*180/torch.pi)
                    # print(phi_delta*180/torch.pi)
                    l1 = np.linalg.norm(coordinates[1] - coordinates[0], ord=2)

                    # length of circular arc outside of the billiard table
                    length_gamma = np.abs(self.mu * phi_delta)

                    # the action action
                    G = - l1 - length_gamma + 1/self.mu*S
                else:
                    G = None
            else:
                phi2 = None
                G = None

            # from shapely.geometry import Point
            # from matplotlib import pyplot as plt
            # Create circle and ellipse objects
            # circle = Point(center).buffer(self.mu)
            # Calculate intersection area using shapely
            # intersection = circle.intersection(self.table.polygon)
            # fig, ax = self.returnmap.plot(phi, theta, show=False)
            # ax.plot(*intersection.exterior.xy)
            # plt.show()

        return phi, phi2, G
