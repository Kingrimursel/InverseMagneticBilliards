import os
import torch

from ml.models import ReLuModel
from dynamics import Orbit

from helper import Training, Minimizer, Diagnostics
from settings import MODELDIR


def training_procedure(num_epochs=100, type="GeneratingFunction", cs="Custom", train_dataset="train50k.npy"):
    training = Training(num_epochs=num_epochs,
                        type=type,
                        cs=cs,
                        train_dataset=train_dataset)
    training.train()
    training.plot_loss()


def minimization_procedure(a, b, n_epochs=100, dir=None, type="GeneratingFunction", cs="Custom"):
    # load model
    filename = os.path.join(MODELDIR, dir, "model.pth")

    generating_function = ReLuModel(input_dim=2, output_dim=1)
    generating_function.load_state_dict(
        torch.load(filename)["model_state_dict"])

    # number of applications of return map
    frequency = (2, 5)

    # initialize an orbit
    orbit = Orbit(a=a,
                  b=b,
                  frequency=frequency,
                  init="random")

    # initialize and execute minimizer
    minimizer = Minimizer(orbit,
                          generating_function.model,
                          n_epochs=n_epochs,
                          frequency=frequency,
                          exact=True)

    # minimize action
    minimizer.minimize()

    # initialize diagnostics
    diagnostics = Diagnostics(orbit=orbit, type=type, cs=cs)

    observed_frequency = diagnostics.frequency()

    print(f"Expected Frequency: (m,n) = {frequency}. Observed Frequency: (m,n) = {observed_frequency}")

    # plot the orbit
    # orbit.plot()

    # plot the minimization loss
    # minimizer.plot()

    # plot the gradient analysis
    diagnostics.derivative(a, b, dir, ord=2)

    # plot whether the reflection_law is satisfied
    # diagnostics.reflection()
