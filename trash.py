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