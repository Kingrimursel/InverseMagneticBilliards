import numpy as np
from functorch import jacfwd, jacrev, vmap
from scipy.optimize import root_scalar
from matplotlib.widgets import Slider


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
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


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
