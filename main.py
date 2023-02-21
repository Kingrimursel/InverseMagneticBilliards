import os
from procedures import training_procedure, minimization_procedure, Training
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

    # training_procedure(num_epochs=100, type=type, cs=cs)
    # generate_dataset(a, b, mu, 10000, "validate10k.npy", cs="Custom", mode="classic", type="GeneratingFunction")

    minimization_procedure(a, b, dir="GeneratingFunction/Custom/2023-02-18")
