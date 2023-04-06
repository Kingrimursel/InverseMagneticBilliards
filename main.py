import os
import torch
from matplotlib import pyplot as plt
from conf import GRAPHICSDIR, TODAY

from procedures import training_procedure, minimization_procedure
from data.generate import generate_dataset
from util import get_approx_type, mkdir


if __name__ == "__main__":
    # table properties
    # a = 2
    # b = 1
    # k = None  # 1/3

    alist = [1, 2, 2]
    blist = [1, 1, 1]
    klist = [None, None, 1/3]
    frequencies = [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
    datelist = ["2023-04-06", "2023-04-04", "2023-04-04"]  # for classic
    # datelist = ["2023-03-13", "2023-03-13", "2023-04-03"]  # for inversemagnetic
    execs = ["minimize"]

    exact_G = False
    exact_deriv = True

    mu = 1/5

    mode = "classic"

    d1s = []
    d2s = []
    tables = []

    for a, b, k, date in zip(alist, blist, klist, datelist):
        assert a >= b, "a must be greater than or equal to b"
        assert a > 0 and b > 0, "a and b must be positive"
        assert k is None or (k > 0 and k < b), f"k={k} not allowd"
        assert k is None or mu <= (b-k)**2/a, "Only high-field implemented"

        if a == b:
            subdir = "circle"
        else:
            if k is None:
                subdir = "ellipse"
            else:
                subdir = "drop"

        tables.append(subdir)

        for exec in execs:
            if exec == "generate":
                generate_dataset(a,
                                 b,
                                 k,
                                 mu,
                                 10000,
                                 "validate10k.npy",
                                 subdir=subdir,
                                 mode=mode)
            elif exec == "train":
                training_procedure(a=a,
                                   b=b,
                                   k=k,
                                   mu=mu,
                                   num_epochs=256,
                                   subdir=subdir,
                                   mode=mode,
                                   train_dataset="train100k.npy",
                                   save=True,
                                   batch_size=512)
            elif exec == "minimize":
                d1s_table = []
                d2s_table = []

                for frequency in frequencies:
                    d1s_freq, d2s_freq = minimization_procedure(
                        a,
                        b,
                        k,
                        mu,
                        exact_G=exact_G,
                        exact_deriv=exact_deriv,
                        show=False,
                        frequency=frequency,
                        helicity="pos",
                        plot_points=True,
                        n_epochs=2000,
                        dir=os.path.join(mode, subdir, date)
                    )

                    d1s_table.append(d1s_freq)
                    d2s_table.append(d2s_freq)

                d1s.append(torch.stack(d1s_table))
                d2s.append(torch.stack(d2s_table))

    d1s = torch.stack(d1s)
    d2s = torch.stack(d2s)

    d1s_mean = torch.mean(d1s, dim=2)
    d1s_std = torch.std(d1s, dim=2)

    if d2s.nelement() != 0:
        d2s_mean = torch.mean(d2s, dim=2)
        d2s_std = torch.std(d2s, dim=2)

    fig, ax = plt.subplots()
    ax.set_xlabel("$m$")
    ax.set_ylabel("Error")

    for i in range(d1s.shape[0]):
        color = next(ax._get_lines.prop_cycler)['color']
        x = [f[0] for f in frequencies]
        ax.errorbar(x,
                    d1s_mean[i],
                    yerr=d1s_std[i],
                    label=tables[i],
                    color=color,
                    fmt=".",
                    ls="solid",
                    elinewidth=5,
                    capsize=5,
                    markeredgewidth=5,
                    alpha=0.5)

        if d2s.nelement() != 0:
            ax.errorbar(x,
                        d2s_mean[i],
                        yerr=d2s_std[i],
                        color=color,
                        fmt=".",
                        ls="dashed",
                        elinewidth=5,
                        capsize=5,
                        markeredgewidth=5,
                        alpha=0.5)

    plt.legend(loc="best")

    img_dir = os.path.join(
        GRAPHICSDIR, mode, get_approx_type(exact_G, exact_deriv), TODAY)
    mkdir(img_dir)
    plt.savefig(os.path.join(img_dir, "errors.png"))

    plt.show()

    plt.close()
