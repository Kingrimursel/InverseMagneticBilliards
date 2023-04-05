import torch
import numpy as np
import numdifftools as nd

from setting import Table
from dynamics import ReturnMap
from util import batch_jacobian, pair, circ_int_params, area_overlap


class DiscreteAction:
    def __init__(self, action_fn, table, exact_deriv=True):
        self.action_fn = action_fn
        self.table = table
        self.exact_deriv = exact_deriv

    def __call__(self, phis):
        action = torch.sum(self.action_fn(pair(phis)))
        return action

    def grad_norm(self, x):
        if self.exact_deriv:
            jac = batch_jacobian(self.__call__, x)
        else:
            jac = batch_jacobian(self.__call__, x, approx=True)

        # if getattr(self.action_fn, "__name__", "") == "exact":
        #    jac = batch_jacobian(self.__call__, x, approx=True)
        # else:
        #    jac = batch_jacobian(self.__call__, x)

        if jac.ndim == 1:
            return torch.linalg.norm(jac)
        else:
            return torch.linalg.norm(jac, dim=1)

    def grad_norm_summed(self, x):
        return torch.sum(self.grad_norm(x))


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
        if self.mode == "classic":
            p0 = self.table.boundary(phis[:, 0])
            p2 = self.table.boundary(phis[:, 1])
            if p0.requires_grad:
                return - torch.linalg.norm(p0 - p2, dim=1)
            else:
                return - torch.from_numpy(np.linalg.norm(p0 - p2, axis=1))

        """
         else:
            Gs = []
            coordinates = self.table.boundary(phis[:, 0])
            coordinates = np.vstack([coordinates, coordinates[0]])
            #coordinates[1] = p1s
            #coordinates = np.vstack((coordinates, coordinates[-1]))

            p1s, center = self.table.get_exit_points(phis[:, 0], self.mu)

            for i in range(len(center)):
                all_coordinates = np.vstack((coordinates[i], p1s[i], coordinates[i+1]))

                _, _, G = self.__call__(None, None, coordinates=all_coordinates, center=center[i])
                Gs.append(G)

         Gs = torch.tensor(Gs)

         return Gs
        """

    def __call__(self, phi, theta):
        # if coordinates are not passed, compute them by using the return map
        coordinates, center = self.returnmap(phi, theta)

        # evaluate the generating function
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
