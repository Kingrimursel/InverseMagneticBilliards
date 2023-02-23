import os
from procedures import training_procedure, minimization_procedure
from data.generate import generate_dataset


if __name__ == "__main__":
    # table properties
    a = 2
    b = 1

    # magnetic properties
    mu = 1/5

    cs = "Custom"
    mode = "classic"
    type = "GeneratingFunction"

    # generate_dataset(a, b, mu, 100000, "train100k.npy", cs="Custom", mode="classic", type="GeneratingFunction")
    #training_procedure(num_epochs=100,
    #                   type=type,
    #                   cs=cs,
    #                   train_dataset="train100k.npy")

    minimization_procedure(a, b, n_epochs=100, dir="GeneratingFunction/Custom/2023-02-23", type=type, cs=cs)
