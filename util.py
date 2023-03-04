import os
import torch

import numpy as np
from pathlib import Path
from shapely.geometry import Point
from shapely import affinity

from functorch import jacfwd, jacrev, vmap, hessian
from scipy.optimize import root_scalar
from conf import GRAPHICSDIR


def get_legend(a, b, m, n):
    return f"Variables\na = {a}\nb = {b}\n\nm = {m}\nn = {n}"


def update_slider_range(m, n, m_slider, n_slider, m_n_max):
    m_min = 1
    m_max = n
    n_min = m
    n_max = m_n_max

    n_slider.valmin = n_min
    n_slider.valmax = n_max
    m_slider.valmin = m_min
    m_slider.valmax = m_max


def rotate_vector(v, theta):
    """
    Rotate vector v by angle theta
    """

    # rotation matrix
    M = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])

    v_rotated = M @ v

    return v_rotated


def get_initial_theta(mu=1/5, m=1, n=1):
    sol = root_scalar(implicit_theta, args=(m, n, mu), x0=np.pi/2,
                      bracket=[0, np.pi], fprime=True, fprime2=True)

    return sol


def get_chi(theta0, mu):
    chi = theta0 + np.arcsin(mu*np.sin(theta0) /
                             np.sqrt(1 + mu**2 - 2*mu*np.cos(theta0)))

    return chi


def implicit_theta(theta0, m, n, mu):
    chi = get_chi(theta0, mu)

    delta = chi - m/n*np.pi

    delta_deriv = 1 + ((m*np.cos(theta0))/np.sqrt(1 + m**2 - 2*m*np.cos(theta0)) - (m**2*np.sin(theta0)**22)/(
        1 + m**2 - 2*m*np.cos(theta0))**(3/2))/np.sqrt(1 - (m**2*np.sin(theta0)**2)/(1 + m**2 - 2*m*np.cos(theta0)))

    delta_deriv_deriv = (m*(-1 + m**2)*(-1 + m*np.cos(theta0))*np.sin(theta0))/((1 + m**2 - 2*m *
                                                                                np.cos(theta0)) ** (5/2)*np.sqrt((-1 + m*np.cos(theta0))**2/(1 + m**2 - 2*m*np.cos(theta0))))

    return delta, delta_deriv, delta_deriv_deriv


def unit_vector(vector):
    if torch.is_tensor(vector):
        return vector / torch.linalg.norm(vector, dim=0)
    else:
        return vector / np.linalg.norm(vector)


def sigmoid_scaled(x, alpha=1):
    return torch.nn.Sigmoid()(alpha*x)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if torch.is_tensor(v1):
        sp = torch.clip((v1_u*v2_u).sum(axis=1), -1.0, 1.0)

        test = torch.arccos(sp)

        return torch.arccos(sp)
    else:
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def pair(x, periodic=True):
    pairs = torch.cat((torch.unsqueeze(x, dim=1),
                       torch.unsqueeze(torch.roll(x, -1), dim=1)), 1)

    if not periodic:
        pairs = pairs[:-1]

    return pairs


def batch_hessian(f, input):
    if input.ndim == 1:
        hes = hessian(f)(input)
    else:
        jac = vmap(hessian(f), in_dims=(0,))(input)

    return jac


def batch_jacobian(f, input):
    """
    Compute the diagonal entries of the jacobian of f with respect to x
    :param f: the function
    :param x: where it is to be evaluated
    :return: diagonal of df/dx. First dimension is the derivative
    """

    # compute vectorized jacobian. For curvature because of nested derivatives, for some of the backward functions
    # the forward mode AD is not implemented
    if input.ndim == 1:
        try:
            jac = jacfwd(f)(input)
        except NotImplementedError:
            jac = jacrev(f)(input)

    else:
        try:
            jac = vmap(jacfwd(f), in_dims=(0,))(input)
        except NotImplementedError:
            jac = vmap(jacrev(f), in_dims=(0,))(input)

    return jac

# use sympy to calculate the area enclosed by two ellipses


def get_polar_angle(a,  p):
    if p[1] >= 0:
        phi = np.arccos(p[0]/a)
    else:
        phi = 2*np.pi - np.arccos(p[0]/a)

    return phi


def intersection_parameters(table, mu, center):
    circle = Point(center).buffer(mu)

    intersection = table.polygon.exterior.intersection(circle.exterior)

    if not intersection.is_empty:
        sol = table.polygon.exterior.intersection(circle.exterior).geoms

        phii = get_polar_angle(
            1, 1/mu*(list(list(sol[0].coords)[0]) - np.array(center)))
        phif = get_polar_angle(
            1, 1/mu*(list(list(sol[1].coords)[0]) - np.array(center)))

        return phii, phif

    else:
        return None, None

    from matplotlib import pyplot as plt
    fig, ax = plt.subplots()

    ax.plot(*circle.exterior.xy)
    ax.plot(*table.polygon.exterior.xy)

    plt.show()

    # sol = np.array(sol).astype(np.float64)

    print(sol)

    # print(sol)
    # print(np.array(center))


def area_overlap(table, mu, center):
    """Returns the area enclosed by an ellipse and a circle

    Args:
        a (int): length of first semi axis
        b (int): length of second semi axis
        center (np.array of shape 2): center of circle
        mu (float): radius of circle

    Returns:
        float: area enclosed by ellipse and circle
    """

    # Create circle and ellipse objects
    circle = Point(center).buffer(mu)

    # Calculate intersection area using shapely
    intersection = circle.intersection(table.polygon)

    return intersection.area


def is_left_of(v, p):
    """ Function that checks if a point p is on the left of a vector v

    Args:
        v (np.array of shape (2)): vector
        p (np.array of shape (2)): point

    Returns:
        bool: True if p is on the left of v, False otherwise
    """

    return np.cross(v, p) > 0


def solve_polynomial(a, b, c, d, e):
    """Solve a fourth order polynomial equation

    Args:
        a (float): coefficient of x^4
        b (float): coefficient of x^3
        c (float): coefficient of x^2
        d (float): coefficient of x^1
        e (float): coefficient of x^0

    Returns:
        x (np.array of shape (4)): roots of the equation
    """

    # find roots
    roots = np.roots(np.array([a, b, c, d, e]))

    # extract the real solutions
    roots = np.real(roots[np.isreal(roots)])

    return roots


def generate_readme(path, content):
    """Generate a readme file with the given content

    Args:
        path (str): path to the folder where the readme file will be generated
        content (str): content of the readme file
    """

    with open(os.path.join(path, "README.md"), "w") as f:
        f.write(content)


def mkdir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)
