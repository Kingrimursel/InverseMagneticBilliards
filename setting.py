import torch
import numpy as np
from shapely.geometry import Point
from shapely import affinity
from scipy.special import ellipeinc
import matplotlib as mpl

from util import solve_polynomial, get_polar_angle
from conf import RES_TABLE


class Table:
    def __init__(self, a=1, b=1):
        """
        An elliptical billiards table with main axes a and b
        """

        self.a = a
        self.b = b

        self.polygon = affinity.scale(
            Point(0, 0).buffer(1, resolution=RES_TABLE), a, b)

    def get_other_collision(self, linestring, p0):
        """
        Calculates the second intersection point of a straight line parametrized by its anchor p
        and a direction v with the billiard table's boundary.
        """

        intersection = self.polygon.intersection(linestring)

        # from matplotlib import pyplot as plt
        # fig, ax = plt.subplots()
        # ax.plot(*self.polygon.exterior.xy)
        # ax.plot(*linestring.xy)
        # ax.set_aspect("equal")
        # plt.show()

        # make sure to return the intersection that is further away from p0
        if len(intersection.coords) == 2:
            if np.linalg.norm(p0 - intersection.coords[0]) > np.linalg.norm(p0 - intersection.coords[1]):
                return np.array(intersection.coords[0])
            else:
                return np.array(intersection.coords[1])
            #return np.array(intersection.coords[1])
        else:
            return np.array([None, None])

        try:
            if np.linalg.norm(intersection.coords[0] - test) >= np.linalg.norm(intersection.coords[1] - test):
                print("ALARM")
        except:
            from matplotlib import pyplot as plt
            fig, ax = plt.subplots()
            ax.plot(*self.polygon.exterior.xy)
            ax.plot(*linestring.xy)
            ax.set_aspect("equal")
            plt.show()

        return np.array(intersection.coords[1])

        # t = (- p[0]*v[0]/self.a**2 - p[1]*v[1]/self.b**2 + np.sqrt(
        #    self.b**2*v[0]**2-p[1]**2*v[0]**2+2*p[0]*p[1]*v[0]*v[1]+self.a**2*v[1]**2 - p[0]**2*v[1]**2)/(self.a*self.b))/(v[0]**2/self.a**2 + v[1]**2/self.b**2)

        # return t

    def boundary(self, phi):
        if torch.is_tensor(phi):
            return torch.stack([self.a*torch.cos(phi), self.b*torch.sin(phi)]).T
        else:
            return np.array([self.a*np.cos(phi), self.b*np.sin(phi)])

    def tangent(self, phi):
        if torch.is_tensor(phi):
            v = torch.stack([-self.a*torch.sin(phi), self.b*torch.cos(phi)])
            v = v/torch.norm(v)
        else:
            v = np.array([-self.a*np.sin(phi), self.b*np.cos(phi)])
            v = v/np.linalg.norm(v)

        return v

    def get_polar_angle(self, p):
        """Get the elliptical polar angle of a point p on the tables boundary.
        CAUTION: cos is not injective on [0, 2*np.pi]

        Args:
            p (np.array of shape (2)): the point

        Returns:
            phi (float): the polar angle
        """

        # res =  np.mod(np.arctan2(point[1]/self.b, point[0]/self.a), 2*np.pi)
        return get_polar_angle(self.a, p)

    def get_patch(self, fill=None):
        return mpl.patches.Ellipse((0, 0), 2*self.a, 2*self.b, fill=fill, alpha=1, facecolor="white", edgecolor="black")

    def get_arclength(self, phi):
        return ellipeinc(phi, 1-(self.a/self.b)**2)

    def get_circumference(self):
        return self.get_arclength(2*np.pi)

    def get_polygon(self):
        return self.polygon

    def get_reenter_point(self, mu, center, exit_point):
        """ Get the point where the particle reenters the table after a bounce

        Args:
            mu (float): center of the larmor circle
            exit_point (np.array of shape (2)): point where the particle leaves the table
        """

        circle = Point(center).buffer(mu)
        intersection = self.polygon.exterior.intersection(circle.exterior)

        if not intersection.is_empty:
            sol = intersection.geoms

            intersections = [list(list(sol[0].coords)[0]),
                             list(list(sol[1].coords)[0])]

            idx_reenter_point = np.argmax(
                np.sum((intersections - exit_point)**2, axis=1))
            reenter_point = intersections[idx_reenter_point]
        else:
            reenter_point = [None, None]

        return reenter_point
