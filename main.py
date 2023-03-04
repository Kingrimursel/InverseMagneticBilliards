import os
from procedures import training_procedure, minimization_procedure
from data.generate import generate_dataset


if __name__ == "__main__":
    # table properties
    a = 1
    b = 1

    # magnetic properties
    mu = 1/5

    cs = "custom"
    mode = "inversemagnetic"
    type = "generatingfunction"
    subdir = "circle"

    exec = "minimize"

    if exec == "generate":
        generate_dataset(a,
                         b,
                         mu,
                         100000,
                         "train100k.npy",
                         cs=cs,
                         subdir=subdir,
                         mode=mode,
                         type=type)
    elif exec == "train":
        training_procedure(a=a,
                           b=b,
                           mu=mu,
                           num_epochs=100,
                           type=type,
                           cs=cs,
                           subdir=subdir,
                           mode=mode,
                           train_dataset="train100k.npy",
                           save=True,
                           batch_size=1024,
                           alpha=1e0)
    elif exec == "minimize":
        minimization_procedure(
            a,
            b,
            mu,
            n_epochs=1000,
            dir="generatingfunction/custom/inversemagnetic/circle/2023-03-03",
            type=type,
            cs=cs,
            mode=mode)
