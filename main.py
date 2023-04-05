import os
from procedures import training_procedure, minimization_procedure
from data.generate import generate_dataset


if __name__ == "__main__":
    # table properties
    a = 2
    b = 1
    k = None  # 1/3

    mu = 1/5

    mode = "classic"

    assert a >= b, "a must be greater than or equal to b"
    assert a > 0 and b > 0, "a and b must be positive"
    assert k is None or (k > 0 and k < b), "k must be positive and less than b"
    assert k is None or mu <= (b-k)**2/a, "Only high-field limit is implemented"


    if a == b:
        subdir = "circle"
    else:
        if k is None:
            subdir = "ellipse"
        else:
            subdir = "drop"

    execs = ["minimize"]
    for exec in execs:
        if exec == "generate":
            generate_dataset(a,
                             b,
                             k,
                             mu,
                             100000,
                             "train100k.npy",
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
            # frequencies = [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]
            frequencies = [(1, 7)]

            for frequency in frequencies:
                minimization_procedure(
                    a,
                    b,
                    k,
                    mu,
                    exact_G=True,
                    exact_deriv=False,
                    show=False,
                    frequency=frequency,
                    helicity="pos",
                    plot_points=True,
                    n_epochs=2000,
                    # dir=os.path.join(mode, subdir, "2023-03-13")
                    # dir=os.path.join(mode, subdir, "2023-04-03")
                    dir=os.path.join(mode, subdir, "2023-04-04")
                )
