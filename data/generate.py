import sys
import os
import torch

from tqdm import tqdm

import numpy as np
from physics import Action
from conf import DATADIR


def generate_dataset(a, b, mu, n_samples, filename, cs="Birkhoff", type="ReturnMap", mode="classic"):
    """Automatically generate a return map dataset

    Args:
        a (int): first semi axis
        b (int): second semi axis
        mu (float): larmor radius
    """

    filename = os.path.join(DATADIR, type, cs, mode, filename)

    eps = 1e-10

    if type == "GeneratingFunction":
        print(f"GENERATING DATASET OF SIZE {n_samples}...")

        # initialize grid of angles
        phis = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        thetas = np.random.uniform(low=eps, high=np.pi-eps, size=n_samples)

        # actually calculate action
        action = Action(a, b, mu, mode=mode, cs=cs)

        phi0s = []
        phi2s = []
        Gs = []

        for phi0, theta0 in tqdm(zip(phis, thetas), total=len(phis)):
            phi0, phi2, G = action(phi0, theta0)

            if phi2 is not None and G is not None:
                phi0s.append(phi0)
                phi2s.append(phi2)
                Gs.append(G)

        # phis = np.vstack((phi0s, phi2s)).T

        print(f"SAVING DATASET TO {filename}...")
        dataset = np.vstack((phi0s, phi2s, Gs)).T

        np.save(filename, dataset)
