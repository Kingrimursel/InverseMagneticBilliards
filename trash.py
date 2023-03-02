# plot remaining parts
i = 0
for p, v in zip(ps, vs):
    # ax.scatter(*p, c="purple", zorder=20)

    #if i % 2 == 0:
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