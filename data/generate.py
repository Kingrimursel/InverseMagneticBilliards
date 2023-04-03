import sys
import os
import torch

from tqdm import tqdm

import numpy as np
from physics import Action
from conf import DATADIR
from util import generate_readme, mkdir


def generate_dataset(a, b, k, mu, n_samples, filename, cs="Birkhoff", type="ReturnMap", mode="classic", subdir=""):
    """Automatically generate a return map dataset

    Args:
        a (int): first semi axis
        b (int): second semi axis
        mu (float): larmor radius
    """

    data_dir = os.path.join(DATADIR, type, cs, mode, subdir)
    mkdir(data_dir)

    filename = os.path.join(data_dir, filename)

    if type == "generatingfunction":
        print(f"GENERATING DATASET OF SIZE {n_samples}...")

        # initialize grid of angles
        # phis = np.linspace(0, 2*np.pi, num=n_samples)
        # thetas = np.linspace(eps, np.pi-eps, num=n_samples)
        phis = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        thetas = np.random.uniform(low=0, high=np.pi, size=n_samples)
        coordinates = np.vstack([phis, thetas]).T

        # actually calculate action
        action = Action(a, b, k, mu, mode=mode, cs=cs)

        phi0s = []
        phi2s = []
        Gs = []

        # xx, yy = np.meshgrid(phis, thetas)
        # coordinates = np.vstack([xx.ravel(), yy.ravel()]).T
        # np.random.shuffle(coordinates)

        for angles0 in tqdm(coordinates, total=n_samples):
            phi0_orig, theta0 = angles0[0], angles0[1]
            #phi0_orig = 0
            #theta0 = np.pi/2 - 0.05
            phi0, phi2, G = action(phi0_orig, theta0)

            # polygonial approximation is not exact
            if phi2 is not None and G is not None:
                # action.returnmap.plot(phi0, theta0)
                phi0s.append(phi0)
                phi2s.append(phi2)
                Gs.append(G)

        print(f"SAVING DATASET TO {filename}...")
        dataset = np.vstack((phi0s, phi2s, Gs)).T

        np.save(filename, dataset)

        # generate readme
        generate_readme(
            data_dir, f"a={a},\nb={b},\nk={k},\nmu={mu},\nn_samples={n_samples}")
