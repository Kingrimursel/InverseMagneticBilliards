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

    dir = os.path.join(type, cs)

    training_procedure(num_epochs=100, reldir=dir)
    # minimization_procedure(a, b,
    #                       rm_filename=os.path.join(os.path.dirname(
    #                           __file__), "output/checkpoints/models/2023-02-17/ReturnMap/model.pth"),
    #                       u_filename=os.path.join(os.path.dirname(__file__), "output/checkpoints/models/2023-02-17/ImplicitU/model.pth"))

    # generate_dataset(a, b, mu, 10000, "validate10k.npy", cs="Custom", mode="classic", type="GeneratingFunction")
