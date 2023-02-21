import os
import torch

from ml.models import ReLuModel
from dynamics import Orbit

from helper import Training, Minimizer, Diagnostics


def training_procedure(num_epochs=100, type="GeneratingFunction", cs="Custom", train_dataset="train50k.npy"):
    training = Training(num_epochs=num_epochs, type=type,
                        cs=cs, train_dataset=train_dataset)
    training.train()
    training.plot_loss()


def minimization_procedure(a, b, dir=None):
    # load model
    filename = os.path.join(os.path.dirname(__file__),
                            "output/models", dir, "model.pth")

    generating_function = ReLuModel(input_dim=2, output_dim=1)
    generating_function.load_state_dict(
        torch.load(filename)["model_state_dict"])

    # number of applications of return map
    n_return = 5

    # number of epochs for minimization
    n_epochs = 200

    # initialize an orbit
    orbit = Orbit(a=a, b=b, n=n_return, init="random")

    # initialize and execute minimizer
    minimizer = Minimizer(orbit, generating_function.model, n_epochs=n_epochs)

    # minimize action
    minimizer.minimize()

    # initialize diagnostics
    diagnostics = Diagnostics(orbit)

    # plot the orbit
    orbit.plot()

    # plot the minimization loss
    minimizer.plot()

    # plot whether the reflection_law is satisfied
    diagnostics.plot()
