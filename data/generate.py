import sys
import os
import torch

import numpy as np
from dynamics import Action, Orbit
from conf import DATADIR


def generate_dataset(a, b, mu, n_samples, filename, cs="Birkhoff", type="ReturnMap", mode="classic"):
    """Automatically generate a return map dataset

    Args:
        a (int): first semi axis
        b (int): second semi axis
        mu (float): larmor radius
    """

    filename = os.path.join(DATADIR, type, cs, filename)

    eps = 1e-10
    if type == "ReturnMap":
        print(f"GENERATING DATASET OF SIZE {n_samples}...")
        
        # initialize grid of angles
        phis = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        thetas = np.random.uniform(low=eps, high=np.pi-eps, size=n_samples)

        orbit = Orbit(a, b, frequency=(1, 1), mode=mode, cs=cs)

        coordinates = []

        for phi, theta in zip(phis, thetas):
            orbit.update(phi, theta)
            new_coordinates = orbit.step(N=1)
            coordinates.append(new_coordinates)

        coordinates = np.stack(coordinates[0::2])

        # print(f"SAVING DATASET TO {filename}...")
        # np.save(filename, coordinates)

    elif type == "GeneratingFunction":
        print(f"GENERATING DATASET OF SIZE {n_samples}...")
        
        # initialize grid of angles
        phi0 = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        theta0 = np.random.uniform(low=eps, high=np.pi-eps, size=n_samples)

        # actually calculate action
        # TODO: instead of returning these, just save them as class variables
        # TODO: plot me, to see whether it works
        action = Action(a, b, mu, mode=mode, cs=cs)
        _, phi2, G = action(phi0, theta0)

        phis = np.vstack((phi0, phi2)).T

        # print(f"SAVING DATASET TO {filename}...")
        # dataset = np.vstack((phi0, phi2, Gs)).T

        # np.save(filename, dataset)
