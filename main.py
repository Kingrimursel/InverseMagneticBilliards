import os
from procedures import training_procedure, minimization_procedure
from data.generate import generate_dataset


if __name__ == "__main__":
    # table properties
    a = 2
    b = 1

    # magnetic properties
    mu = 1/5

    cs = "custom"
    mode = "inversemagnetic"
    type = "generatingfunction"
    subdir = "ellipse"

    execs = ["minimize"]
    for exec in execs:
        if exec == "generate":
            generate_dataset(a,
                             b,
                             mu,
                             10000,
                             "validate10k.npy",
                             cs=cs,
                             subdir=subdir,
                             mode=mode,
                             type=type)
        elif exec == "train":
            training_procedure(a=a,
                               b=b,
                               mu=mu,
                               num_epochs=256,
                               type=type,
                               cs=cs,
                               subdir=subdir,
                               mode=mode,
                               train_dataset="train100k.npy",
                               save=True,
                               batch_size=512)
        elif exec == "minimize":
            frequencies = [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7)]

            for frequency in frequencies:
                minimization_procedure(
                    a,
                    b,
                    mu,
                    exact=False,
                    show=False,
                    frequency=frequency,
                    helicity="pos",
                    n_epochs=2000,
                    dir=os.path.join(type, cs, mode, subdir, "2023-03-13")
                )
