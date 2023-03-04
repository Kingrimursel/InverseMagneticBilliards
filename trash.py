# plot remaining parts
from shapely.geometry import Point
import matplotlib.pyplot as plt
from shapely import affinity
i = 0
for p, v in zip(ps, vs):
    # ax.scatter(*p, c="purple", zorder=20)

    # if i % 2 == 0:
    #    ax.plot([ps[i][0], ps[i+1][0]], [ps[i][1], ps[i+1][1]], c="navy")
    # else:
    #    v_to_mu = mu * rotate_vector(v, np.pi/2)
    #    larmor_center = p + v_to_mu
    #    ax.scatter(*larmor_center, c="yellow", zorder=20)

    #     circle = plt.Circle(tuple(larmor_center), mu,
    #                         fill=None, alpha=1, edgecolor="navy", zorder=0)
    #     ax.add_patch(circle)

    i += 1


def test_circ(xi, xf):
    return y0*(xf-xi) + 1/2*((x0 - xf)*abs(y0 - yf) + (xi - x0)*abs(y0 - yi)) + mu**2*np.angle(1/mu**2*((x0-xi)*(x0-xf) + abs(y0-yi)*abs(y0-yf) + 1j*((x0-xf)*abs(y0-yi) - (x0 - xi)*abs(y0-yf))))


def int_ellipse(x):
    return b*(1/2*x*np.sqrt(1-(x/a)**2) - np.log(-np.sqrt(np.array([-1/a**2], dtype=complex))*x + np.sqrt(1 - (x/a)**2))/(2*np.sqrt(np.array([-1/a**2], dtype=complex))))
    # return b*(1/2*x*np.sqrt(1 - x**2/a**2) - np.log(-x*np.sqrt(-1j/a**2) + np.sqrt(1 - x**2/a**2))/(2*np.sqrt(-1j/a**2)))


def int_circ(x):
    return -1j*mu**2*np.log(-1j*x + 1j*center[0] + np.sqrt(mu**2 - (x - center[0])**2))/2 + x*center[1] - x*np.sqrt(mu**2 - (x - center[0])**2)/2 + center[0]*np.sqrt(mu**2 - (x - center[0])**2)/2


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


class Trajectory:
    def __init__(self, phi0, theta0, mu, a, b, mode="classic", cs="Birkhoff"):
        """Trajectory of a charged particle

        Args:
            phi0 (_type_): _description_
            theta0 (_type_): _description_
            mu (_type_): _description_
            a (_type_): _description_
            b (_type_): _description_
            mode (str, optional): If set to 'classic', normal billiards. Else, inverse magnetic billiards. Defaults to "classic".
        """
        # defining the billiards table
        self.table = Table(a=a, b=b)
        self.a = a
        self.b = b

        self.mode = mode
        self.cs = cs

        if self.mode != "classic":
            chi = get_chi(theta0, mu)

        # initial conditions
        s0 = self.table.get_arclength(phi0)

        t0 = self.table.tangent(phi0)
        v0 = rotate_vector(t0, theta0)

        self.phi0 = phi0
        self.theta = theta0
        self.s0 = s0

        # constants
        self.mu = mu
        if self.mode != "classic":
            self.theta = theta0
            self.chi = chi

        # initialize runtime variables
        self.s = s0
        self.u = -np.cos(theta0)
        self.phi = phi0
        self.v = v0
        self.p = self.table.boundary(phi0)
        self.n_step = 0

    def update(self, phi0, theta0):
        self.phi = phi0
        self.theta = theta0

        self.v = rotate_vector(self.table.tangent(phi0), theta0)
        self.p = self.table.boundary(phi0)
        self.s = self.table.get_arclength(phi0)
        self.u = -np.cos(theta0)

    def step(self, N=1):
        """
        Do n iterations of the return map.
        """

        vs = [self.v]
        ps = [self.p]

        coordinates = [[self.s, self.u]]

        while N > 0:
            if self.mode == "classic":
                # get time of collision with boundary
                t = self.table.get_collision(self.p, self.v)

                # get collision point
                self.p = self.p + t*self.v

                # ellipse parameter corresponding to collision point. Caution: cos is not injective!
                if self.p[1] >= 0:
                    phi1 = np.arccos(self.p[0]/self.table.a)
                else:
                    phi1 = 2*np.pi - np.arccos(self.p[0]/self.table.a)

                # caculate arclength
                s1 = self.table.get_arclength(phi1)

                # update runtime variables
                theta1 = angle_between(self.v, self.table.tangent(phi1))
                u1 = - np.cos(theta1)

                self.v = rotate_vector(self.table.tangent(phi1), theta1)
                self.s = s1
                self.u = u1
                self.phi = phi1

                if self.cs == "Birkhoff":
                    coordinates.append([s1, u1])
                elif self.cs == "custom":
                    coordinates.append([phi1, theta1])
                else:
                    return
            else:
                # corresponds to a chord
                if self.n_step % 2 == 0:
                    # get time of collision, the parameter of the straight line
                    t = self.table.get_collision(self.p, self.v)

                    # get collision point
                    self.p = self.p + t*self.v

                # corresponds to a magnetic arc
                else:
                    # the direction of the l_2 chord
                    v_chord = rotate_vector(self.v, self.chi)
                    # intersection of l_2 with the boundary
                    t = self.table.get_collision(self.p, v_chord)
                    # move base point along l_2 chord
                    self.p = self.p + t*v_chord
                    # rotate the velocity by psi=2*chi
                    self.v = rotate_vector(self.v, 2*self.chi)

            self.n_step += 1
            N -= 1

            ps.append(self.p)
            vs.append(self.v)

        ps = np.stack(ps)
        vs = np.stack(vs)
        coordinates = np.stack(coordinates)

        if self.mode == "classic":
            return coordinates
        else:
            return ps, vs

    def plot(self, ax, N=10, legend=None):
        ps, vs = self.step(N=N)

        # plot billiards table
        ax.add_patch(self.table.get_patch(fill="white"))

        # plot exit and reentry points
        ax.scatter(ps[:, 0], ps[:, 1], c="purple", zorder=20)

        # plot the larmor centers and -circles
        larmor_centers = ps[1::2] + self.mu * \
            rotate_vector(vs[1::2].T, np.pi/2).T
        ax.scatter(larmor_centers[:, 0],
                   larmor_centers[:, 1], c="yellow", zorder=20)

        circles = PatchCollection([plt.Circle(tuple(larmor_center), self.mu, alpha=1,
                                              edgecolor="navy", zorder=0) for larmor_center in larmor_centers])

        circles.set_facecolor([0, 0, 0, 0])
        circles.set_edgecolor([0, 0, 0, 1])
        circles.set_zorder(0)
        ax.add_collection(circles)

        # plot the non-magnetic chords
        xx = np.vstack([ps[0::2][:, 0], ps[1::2][:, 0]])
        yy = np.vstack([ps[0::2][:, 1], ps[1::2][:, 1]])

        ax.plot(xx, yy, c="black")

        # plot trajectory properties
        if legend:
            text_box = AnchoredText(legend, frameon=True, loc=4, pad=0.5)
            plt.setp(text_box.patch, facecolor='white', alpha=0.5)
            ax.add_artist(text_box)


if type == "ReturnMap":
    print(f"GENERATING DATASET OF SIZE {n_samples}...")

    # initialize grid of angles
    phis = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
    thetas = np.random.uniform(low=eps, high=np.pi-eps, size=n_samples)

    orbit = Orbit(a, b, frequency=(1, 1), mode=mode, cs=cs)

    coordinates = []

    for phi, theta in zip(phis, thetas):
        orbit.update(phi, theta)
        new_coordinates = orbit.step(N=1)
        coordinates.append(new_coordinates)

    coordinates = np.stack(coordinates[0::2])

    # print(f"SAVING DATASET TO {filename}...")
    # np.save(filename, coordinates)


def periodic_orbits(a, b, mu):
    from trash import Trajectory
    # orbit properties
    m = 5
    n = 8
    N = 2*n-1

    # initial conditions
    s0 = 0
    theta0 = get_initial_theta(mu=mu, m=m, n=n).root

    # Plot dynamics
    fig, ax = plt.subplots()
    ax.axis("equal")
    ax.axis("off")

    trajectory = Trajectory(s0, theta0, mu, a=a, b=b, mode="InverseMagnetic")
    trajectory.plot(ax, N=N, legend=get_legend(a, b, m, n))

    m_n_max = 16

    # Slider
    m_slider_ax = fig.add_axes([0.25, 0.1, 0.65, 0.03])
    m_slider = Slider(
        ax=m_slider_ax,
        valstep=np.arange(1, m_n_max),
        label='m',
        valmin=1,
        valmax=m_n_max,
        valinit=m,
    )

    n_slider_ax = fig.add_axes([0.25, 0.05, 0.65, 0.03])
    n_slider = Slider(
        ax=n_slider_ax,
        valstep=np.arange(1, m_n_max),
        label='n',
        valmin=1,
        valmax=m_n_max,
        valinit=n,
    )

    update_slider_range(m, n, m_slider, n_slider, m_n_max)

    def update(val):
        # orbit properties
        m = m_slider.val
        n = n_slider.val
        N = 2*n-1

        # initial conditions
        s0 = 0
        theta0 = get_initial_theta(mu=mu, m=m, n=n).root

        # Plot dynamics
        ax.clear()
        ax.axis("equal")
        ax.axis("off")

        traj = Trajectory(s0, theta0, mu, a=a, b=b)
        traj.plot(ax, N=N, legend=get_legend(a, b, m, n))

        # update slider
        update_slider_range(m, n, m_slider, n_slider, m_n_max)

        fig.canvas.draw()

    m_slider.on_changed(update)
    n_slider.on_changed(update)

    plt.show()
    plt.savefig("figure.png")
