import os
import torch
from pathlib import Path

from ml.models import ReLuModel
from dynamics import Orbit

from helper import Training, Minimizer, Diagnostics
from conf import MODELDIR, GRAPHICSDIR, TODAY
from physics import Action
from setting import Table
from util import batch_jacobian, get_todays_graphics_dir, mkdir, grad


def training_procedure(**kwargs):
    training = Training(**kwargs)
    training.train()
    training.plot_loss()
    training.generate_readme(kwargs.get("a"), kwargs.get("b"), kwargs.get(
        "mu"), kwargs.get("num_epochs"), kwargs.get("batch_size"))


def minimization_procedure(a, b, mu, n_epochs=100, dir=None, helicity="pos", exact=False, frequency=(1, 1), show=True):
    # load model
    filename = os.path.join(MODELDIR, dir, "model.pth")

    type = dir.split("/")[-5]
    cs = dir.split("/")[-4]
    mode = dir.split("/")[-3]
    subdir = dir.split("/")[-2]

    print(f"LOADING model from {filename}")
    G_hat = ReLuModel(input_dim=2, output_dim=1)
    G_hat.load_state_dict(torch.load(filename)["model_state_dict"])

    # choose generating fn
    if exact:
        if mode != "classic":
            print("ERROR: Exact action only available for classic mode")
            exit(1)
        G = Action(a, b, mu, mode=mode, cs=cs).exact
    else:
        G = G_hat


    assert frequency[0] < frequency[1], "m must be less than n"

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
                          G,
                          n_epochs=n_epochs,
                          frequency=frequency)

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
    img_dir = get_todays_graphics_dir(type, cs, mode, subdir, add=str(frequency))

    if mode == "classic":
        img_dir = os.path.join(img_dir, "exact" if exact else "approx")
        mkdir(img_dir)

    orbit.plot(img_dir=img_dir, show=show)

    # diagnostics.landscape(grad(G_hat, norm=True), n=150)
    diagnostics.landscape(G, n=150, img_dir=img_dir, show=show)

    # plot the minimization loss
    minimizer.plot(img_dir=img_dir, show=show)

    # plot the gradient analysis
    # diagnostics.derivative(dir)

    # plot whether the reflection_law is satisfied
    diagnostics.physics(img_dir=img_dir, show=show)
