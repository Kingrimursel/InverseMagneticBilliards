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
    mode = "classic"
    type = "generatingfunction"
    subdir = "ellipse"
    
    exec = "minimize"

    if exec == "generate":
        generate_dataset(a,
                         b,
                         mu,
                         10000,
                         "vaidate10k.npy",
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
                           batch_size=256,
                           alpha=1e0)
    elif exec == "minimize":
        minimization_procedure(
            a,
            b,
            mu,
            n_epochs=2000,
            dir="generatingfunction/custom/inversemagnetic/circle/2023-03-03"
        )
