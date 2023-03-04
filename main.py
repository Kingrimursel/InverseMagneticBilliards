import os
from procedures import training_procedure, minimization_procedure
from data.generate import generate_dataset


if __name__ == "__main__":
    # table properties
    a = 1
    b = 1

    # magnetic properties
    mu = 1/5

    cs = "Custom"
    mode = "inversemagnetic"
    type = "GeneratingFunction"

    exec = "minimize"

    if exec == "generate":
        generate_dataset(a,
                         b,
                         mu,
                         100000,
                         "train100k.npy",
                         cs=cs,
                         mode=mode,
                         type=type)
    elif exec == "train":
        training_procedure(num_epochs=100,
                           type=type,
                           cs=cs,
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
            n_epochs=300,
            dir="GeneratingFunction/Custom/inversemagnetic/2023-03-03",
            type=type,
            cs=cs,
            mode=mode)
