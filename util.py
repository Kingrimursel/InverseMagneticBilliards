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

    xi, yi, xf, yf = *sol[0], *sol[1]

    xi = np.array(xi).astype(np.float64)
    yi = np.array(yi).astype(np.float64)
    xf = np.array(xf).astype(np.float64)
    yf = np.array(yf).astype(np.float64)

    x0 = center[0]
    y0 = center[1]

    def aue(xi, xf):
        integral = 1/(2*a)*b*(xf*np.sqrt(a**2 - xf**2) - xi*np.sqrt(a**2 - xi**2) + 1j*a **
                              2*(np.log(-1j*xf + np.sqrt(a**2 - xf**2)) - np.log(-1j*xi + np.sqrt(a**2 - xi**2))))

        return np.real(integral)

    def auc(xi, xf):
        """Area under ellipse, between the x values xi and xf

        Args:
            xi (float): first x value
            xf (float): second x value

        Returns:
            float: area under ellipse
        """
        integral = 1j/2*mu**2*(np.log(1j*(x0 - xi) + abs(y0 - yi)) - np.log(1j*(x0 - xf) + abs(
            y0 - yf))) + 1/2*((x0 - xf)*abs(y0 - yf) + (xi-x0)*abs(y0 - yi)) + y0*(xf - xi)

        return np.real(integral)

    # function that uses shapely to calculate the area enclosed by an ellipse and a circle

    from shapely.geometry import Point
    from shapely import affinity
    import matplotlib.pyplot as plt
    def ellipse_circle_intersection_area(a, b, center, mu):
        # Create Shapely geometry objects for the ellipse and circle
        ellipse = affinity.scale(Point(0, 0).buffer(1), a, b)
        circle = Point(*center).buffer(mu)

        fig, ax = plt.subplots() 
        ax.plot(*ellipse.exterior.xy)
        ax.plot(*circle.exterior.xy)
        ax.set_aspect("equal")
        plt.show()

        # Calculate the intersection between the two shapes
        intersection = ellipse.intersection(circle)
        
        # If there is no intersection, return 0
        if intersection.is_empty:
            return 0
        
        # If the intersection is a point, return 0 (since a point has zero area)
        if intersection.geom_type == 'Point':
            return 0
        
        # If the intersection is a line, return 0 (since a line has zero area)
        if intersection.geom_type == 'LineString':
            return 0
        
        # If the intersection is a polygon, calculate its area and return it
        if intersection.geom_type == 'Polygon':
            return intersection.area


    def area_difference(xi, xf):

        # use shapely to calculate the area enclosed by an ellipse and a circle
        print(ellipse_circle_intersection_area(a, b, center, mu))

        # return aue(xi, xf) - auc(xi, xf)

    res = area_difference(xi, xf)

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
