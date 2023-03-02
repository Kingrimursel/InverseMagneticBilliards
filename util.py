import torch

import numpy as np
import sympy as sp


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


def area_overlap(a, b, mu, center):
    """Returns the area enclosed by an ellipse and a circle

    Args:
        a (int): length of first semi axis
        b (int): length of second semi axis
        center (np.array of shape 2): center of circle
        mu (float): radius of circle

    Returns:
        float: area enclosed by ellipse and circle
    """

    x_symb, y_symb = sp.symbols('x y', real=True)

    ellipse = (x_symb**2/a**2 + y_symb**2/b**2 - 1)
    circle = ((x_symb-center[0])**2 + (y_symb-center[1])**2 - mu**2)

    sol = sp.solve([ellipse, circle], [x_symb, y_symb])

    xi, _, xf, _ = *sol[0], *sol[1]

    xi = np.array(xi).astype(np.float64)
    xf = np.array(xf).astype(np.float64)

    def int_ellipse(x):
        return b*(x*np.sqrt(1 - x**2/a**2)/2 - np.log(-x*np.sqrt(-1j/a**2) + np.sqrt(1 - x**2/a**2))/(2*np.sqrt(-1j/a**2)))

    def int_circ(x):
        return -1j*mu**2*np.log(-1j*x + 1j*center[0] + np.sqrt(mu**2 - (x - center[0])**2))/2 + x*center[1] - x*np.sqrt(mu**2 - (x - center[0])**2)/2 + center[0]*np.sqrt(mu**2 - (x - center[0])**2)/2

    def area_difference(xi, xf):
        aue = int_ellipse(xf) - int_ellipse(xi)
        auc = int_circ(xf) - int_circ(xi)

        print(aue, auc)

        # print(xi, xf, mu, center[0], center[1])
        # auc = sp.integrate(-sp.sqrt(mu**2 -
        #                   (x-center[0])**2) + center[1], (x, xi, xf))
        return aue - auc

    res = sp.N(area_difference(xi, xf))

    return res


def is_left_of(v, p):
    """ Function that checks if a point p is on the levt of a vector v

    Args:
        v (np.array of shape (2)): vector
        p (np.array of shape (2)): point

    Returns:
        bool: True if p is on the left of v, False otherwise
    """

    return np.cross(v, p) < 0


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
