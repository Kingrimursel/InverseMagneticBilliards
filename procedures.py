import os
import torch
from pathlib import Path

from ml.models import ReLuModel
from dynamics import Orbit

from helper import Training, Minimizer, Diagnostics
from conf import MODELDIR, GRAPHICSDIR, TODAY
from util import mkdir


def training_procedure(**kwargs):
    training = Training(**kwargs)
    training.train()
    training.plot_loss()
    training.generate_readme(kwargs.get("a"), kwargs.get("b"), kwargs.get("mu"), kwargs.get("num_epochs"), kwargs.get("batch_size"))

def minimization_procedure(a, b, mu, n_epochs=100, dir=None, type="generatingfunction", cs="custom", mode="classic"):
    # load model
    filename = os.path.join(MODELDIR, dir, "model.pth")

    generating_function = ReLuModel(input_dim=2, output_dim=1)
    generating_function.load_state_dict(
        torch.load(filename)["model_state_dict"])

    # number of applications of return map
    frequency = (3, 7)

    # initialize an orbit
    orbit = Orbit(a=a,
                  b=b,
                  mu=mu,
                  frequency=frequency,
                  mode=mode,
                  init="uniform")

    # initialize and execute minimizer
    minimizer = Minimizer(orbit,
                          generating_function.model,
                          n_epochs=n_epochs,
                          frequency=frequency,
                          exact=False)

    # minimize action
    minimizer.minimize()

    # initialize diagnostics
    diagnostics = Diagnostics(orbit=orbit, type=type, cs=cs, mode=mode)

    observed_frequency = diagnostics.frequency()

    print(
        f"Expected Frequency: (m, n) = {frequency}. Observed Frequency: (m, n) = {observed_frequency}")

    # plot the orbit
    mkdir(os.path.join(GRAPHICSDIR, type, cs, TODAY))
    img_path = os.path.join(GRAPHICSDIR, type, cs, TODAY, "orbit.png")
    orbit.plot(img_path=img_path)

    # plot the minimization loss
    minimizer.plot()

    # plot the gradient analysis
    # diagnostics.derivative(a, b, dir)

    # plot whether the reflection_law is satisfied
    # diagnostics.reflection()
