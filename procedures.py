import os
import torch
from pathlib import Path

from ml.models import ReLuModel
from dynamics import Orbit

from helper import Training, Minimizer, Diagnostics
from conf import MODELDIR, GRAPHICSDIR, TODAY
from util import batch_jacobian, mkdir, grad


def training_procedure(**kwargs):
    training = Training(**kwargs)
    training.train()
    training.plot_loss()
    training.generate_readme(kwargs.get("a"), kwargs.get("b"), kwargs.get(
        "mu"), kwargs.get("num_epochs"), kwargs.get("batch_size"))


def minimization_procedure(a, b, mu, n_epochs=100, dir=None):
    # load model
    filename = os.path.join(MODELDIR, dir, "model.pth")

    type = dir.split("/")[-5]
    cs = dir.split("/")[-4]
    mode = dir.split("/")[-3]
    subdir = dir.split("/")[-2]

    G_hat = ReLuModel(input_dim=2, output_dim=1)
    G_hat.load_state_dict(
        torch.load(filename)["model_state_dict"])

    # number of applications of return map
    frequency = (1, 3)

    # initialize an orbit
    orbit = Orbit(a=a,
                  b=b,
                  mu=mu,
                  frequency=frequency,
                  mode=mode,
                  init="uniform")

    # initialize and execute minimizer
    minimizer = Minimizer(a,
                          b,
                          orbit,
                          G_hat.model,
                          n_epochs=n_epochs,
                          frequency=frequency,
                          exact=False)

    # minimize action
    minimizer.minimize()

    # initialize diagnostics
    diagnostics = Diagnostics(a,
                              b,
                              mu,
                              orbit=orbit,
                              type=type,
                              cs=cs,
                              mode=mode,
                              subdir=subdir)

    # check if the frequency is correct
    observed_frequency = diagnostics.frequency()

    print(
        f"Expected Frequency: (m, n) = {frequency}. Observed Frequency: (m, n) = {observed_frequency}")

    # plot the orbit
    mkdir(os.path.join(GRAPHICSDIR, type, cs, mode, subdir, TODAY))
    img_path = os.path.join(GRAPHICSDIR, type, cs, mode,
                            subdir, TODAY, "orbit.png")

    # orbit.plot(img_path=img_path)

    diagnostics.landscape(grad(G_hat, norm=True), n=150)

    # plot the minimization loss
    # minimizer.plot()

    # plot the gradient analysis
    # diagnostics.derivative(dir)

    # plot whether the reflection_law is satisfied
    diagnostics.physics()
