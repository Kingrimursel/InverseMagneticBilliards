import sys
import os
import torch

import numpy as np
from dynamics import Trajectory, Action
from setting import Table


def generate_dataset(a, b, mu, n_samples, filename, cs="Birkhoff", type="ReturnMap", mode="classic"):
    """Automatically generate a return map dataset

    Args:
        a (int): first semi axis
        b (int): second semi axis
        mu (float): larmor radius
    """

    filename = os.path.join(os.path.dirname(__file__),
                            "raw", type, cs, filename)

    if type == "ReturnMap":
        offset = 1e-7
        phis = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        thetas = np.random.uniform(
            low=offset, high=np.pi-offset, size=n_samples)

        trajectory = Trajectory(
            0, 0, mu, a=a, b=b, mode="classic", cs=cs)

        coordinates = []

        print(f"GENERATING DATASET OF SIZE {n_samples}...")
        for phi, theta in zip(phis, thetas):
            trajectory.update(phi, theta)
            new_coordinates = trajectory.step(N=1)

            coordinates.append(new_coordinates)

        coordinates = np.stack(coordinates)

        print(f"SAVING DATASET TO {filename}...")
        np.save(filename, coordinates)

    elif type == "GeneratingFunction":
        phi0s = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        phi1s = np.random.uniform(low=0, high=2*np.pi, size=n_samples)

        print(f"GENERATING DATASET OF SIZE {n_samples}...")
        action = Action(a, b, mode=mode)
        Gs = action(phi0s, phi1s)

        phis = np.vstack((phi0s, phi1s)).T

        print(f"SAVING DATASET TO {filename}...")
        dataset = np.vstack((phi0s, phi1s, Gs)).T
        np.save(filename, dataset)
