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
    mode = "birkhoff"
    type = "GeneratingFunction"

    exec = "generate"

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
                           train_dataset="train100k.npy",
                           save=True,
                           batch_size=1024,
                           alpha=1e0)
    elif exec == "minimize":
        minimization_procedure(
            a,
            b,
            n_epochs=100,
            dir="GeneratingFunction/Custom/2023-02-24",  # 23
            type=type,
            cs=cs)
    else:
        from util import area_overlap

        test = area_overlap(2, 1, 0, 1, 0.2)

        print(test)
