import torch
import numpy as np
import matplotlib as mpl
from scipy.special import ellipeinc


class Table:
    def __init__(self, a=1, b=1):
        """
        An elliptical billiards table with main axes a and b
        """

        self.a = a
        self.b = b

    def get_collision(self, p, v):
        """
        Calculates the second intersection point of a straight line parametrized by its anchor p
        and a direction v with the billiard table's boundary.
        """

        t = (- p[0]*v[0]/self.a**2 - p[1]*v[1]/self.b**2 + np.sqrt(
            self.b**2*v[0]**2-p[1]**2*v[0]**2+2*p[0]*p[1]*v[0]*v[1]+self.a**2*v[1]**2 - p[0]**2*v[1]**2)/(self.a*self.b))/(v[0]**2/self.a**2 + v[1]**2/self.b**2)

        return t

    def boundary(self, phi):
        if torch.is_tensor(phi):
            return torch.stack([self.a*torch.cos(phi), self.b*torch.sin(phi)]).T
        else:
            return np.array([self.a*np.cos(phi), self.b*np.sin(phi)])

    def tangent(self, phi):
        if torch.is_tensor(phi):
            return torch.stack([-self.a*torch.sin(phi), self.b*torch.cos(phi)])
        else:
            return np.array([-self.a*np.sin(phi), self.b*np.cos(phi)])

    def get_patch(self, fill=None):
        return mpl.patches.Ellipse((0, 0), 2*self.a, 2*self.b, fill=fill, alpha=1, facecolor="white", edgecolor="black")

    def get_arclength(self, phi):
        return ellipeinc(phi, 1-(self.a/self.b)**2)

    def get_circumference(self):
        return self.get_arclength(2*np.pi)
