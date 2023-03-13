import os
import torch
from pathlib import Path

from ml.models import ReLuModel
from dynamics import Orbit

from helper import Training, Minimizer, Diagnostics
from conf import MODELDIR, GRAPHICSDIR, TODAY
from util import batch_jacobian, get_todays_graphics_dir, mkdir, grad


def training_procedure(**kwargs):
    training = Training(**kwargs)
    training.train()
    training.plot_loss()
    training.generate_readme(kwargs.get("a"), kwargs.get("b"), kwargs.get(
        "mu"), kwargs.get("num_epochs"), kwargs.get("batch_size"))


def minimization_procedure(a, b, mu, n_epochs=100, dir=None, helicity="pos"):
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
    frequency = (2, 5)

    # initialize an orbit
    orbit = Orbit(a=a,
                  b=b,
                  mu=mu,
                  frequency=frequency,
                  mode=mode,
                  helicity=helicity,
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
    img_dir = get_todays_graphics_dir(type, cs, mode, subdir)
    orbit.plot(img_dir=img_dir)

    # diagnostics.landscape(grad(G_hat, norm=True), n=150)
    diagnostics.landscape(G_hat, n=150, img_dir=img_dir)

    # plot the minimization loss
    minimizer.plot(img_dir=img_dir)

    # plot the gradient analysis
    # diagnostics.derivative(dir)

    # plot whether the reflection_law is satisfied
    diagnostics.physics(img_dir=img_dir)
