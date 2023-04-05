import torch
import numpy as np
from shapely.geometry import Point, Polygon
from shapely import affinity
from scipy.special import ellipeinc
import matplotlib as mpl

from util import solve_polynomial, get_polar_angle, get_tangent, is_left_of
from conf import RES_TABLE, RES_LCIRC


class Table:
    def __init__(self, a=1, b=1, k=None):
        """
        An elliptical billiards table with main axes a and b
        """

        self.a = a
        self.b = b
        self.k = k

        self.polygon = self.generate_polygon()

    def generate_polygon(self):
        if self.k is None:
            polygon = affinity.scale(Point(0, 0).buffer(
                1, resolution=RES_TABLE), self.a, self.b)
            
        else:
            n_samples = 4001

            polygon = Polygon([tuple(coord) for coord in self.boundary(np.linspace(0, 2*np.pi, n_samples))])

        return polygon

    def drop(self, t):
        return (self.a*np.cos(t), self.b*np.sin(t) + self.k/2*np.sin(2*t))

    def get_other_collision(self, linestring, p0):
        """
        Calculates the second intersection point of a straight line parametrized by its anchor p
        and a direction v with the billiard table's boundary.
        """

        intersection = self.polygon.intersection(linestring)

        # make sure to return the intersection that is further away from p0
        if len(intersection.coords) == 2:
            if np.linalg.norm(p0 - intersection.coords[0]) > np.linalg.norm(p0 - intersection.coords[1]):
                return np.array(intersection.coords[0])
            else:
                return np.array(intersection.coords[1])
        else:
            return np.array([None, None])

    def boundary(self, phi):
        if self.k is None:
            if torch.is_tensor(phi):
                return torch.stack([self.a*torch.cos(phi), self.b*torch.sin(phi)]).T
            else:
                # TODO: is this ever called? I think this returns the wrong shape!!
                return np.array([self.a*np.cos(phi), self.b*np.sin(phi)]).T
        else:
            if torch.is_tensor(phi):
                return torch.stack([self.a*torch.cos(phi), self.b*torch.sin(phi) + self.k/2*torch.sin(2*phi)]).T
            else:
                return np.stack([self.a*np.cos(phi), self.b*np.sin(phi) + self.k/2*np.sin(2*phi)]).T

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
    

    """
    def get_exit_points(self, phi2, mu):
        ""Get the exit points and the corresponding Larmor centers of an orbit given its reenter points

        Returns:
            np.array, np.array: The exit points and the centers
        ""

        # get reenter points and append the first one to the end because of periodicity
        p2s = self.boundary(phi2)
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
            vertices = Point(p2).buffer(mu).exterior.coords

            # loop over all vertices, checking which corresponding larmor circle comes closest to the real one
            for vertex in vertices:
                vertex = np.array(vertex)

                # get larmor circle
                # TODO: implement base class get_other_point and then implement get_reenter_point using that
                lcirc = Point(vertex).buffer(mu, resolution=RES_LCIRC)

                # obtain exit point corresponding to the given reenter point and larmor circle
                p1 = self.get_reenter_point(mu, vertex, p2)

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
                    p1, lcirc, factor=6*max(self.a, self.b))

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
    """
