import torch
import numpy as np

from setting import Table
from dynamics import ReturnMap
from util import pair, intersection_parameters, area_overlap


class DiscreteAction:
    def __init__(self, action_fn, table, exact=False, *args, **kwargs):
        self.action_fn = action_fn
        self.table = table
        self.exact = exact

    def __call__(self, phis):
        if self.exact:
            points = self.table.boundary(phis)
            if torch.is_tensor(points):
                points2 = torch.roll(points, 1, 0)
                dists = torch.norm(points - points2, dim=1)
                action = torch.sum(dists)
            else:
                points2 = np.roll(points, 1, 0)
                dists = np.linalg.norm(points - points2, axis=1)
                action = np.sum(dists)
        else:
            action = torch.sum(self.action_fn(pair(phis)))
        return action


class Action:
    def __init__(self, a, b, mu, mode="classic", *args, **kwargs):
        self.mode = mode
        self.a = a
        self.b = b
        self.mu = mu
        self.kwargs = kwargs

        self.table = Table(a=a, b=b)

        self.returnmap = ReturnMap(self.a, self.b, self.mu,
                                   frequency=(1, 1), mode=self.mode, **self.kwargs)

    def __call__(self, phi, theta):
        coordinates, center = self.returnmap(phi, theta)

        if self.mode == "classic":
            if coordinates[1][0] is not None:
                G = np.linalg.norm(coordinates[0] - coordinates[1], ord=2)
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

                phii, phif = intersection_parameters(
                    self.table, self.mu, center)

                phi_delta = phif - \
                    phii if phif > phii else (phif + 2*np.pi) - phii

                assert phi_delta > 0

                if phii is not None:
                    # length of first chord
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

        return phi, phi2, G
