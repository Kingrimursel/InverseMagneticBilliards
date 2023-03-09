import sys
import os
import torch

from tqdm import tqdm

import numpy as np
from physics import Action
from conf import DATADIR
from util import generate_readme, mkdir


def generate_dataset(a, b, mu, n_samples, filename, cs="Birkhoff", type="ReturnMap", mode="classic", subdir=""):
    """Automatically generate a return map dataset

    Args:
        a (int): first semi axis
        b (int): second semi axis
        mu (float): larmor radius
    """

    data_dir = os.path.join(DATADIR, type, cs, mode, subdir)
    mkdir(data_dir)

    filename = os.path.join(data_dir, filename)

    eps = 1e-10

    if type == "generatingfunction":
        print(f"GENERATING DATASET OF SIZE {n_samples*n_samples}...")

        # initialize grid of angles
        # phis = np.linspace(0, 2*np.pi, num=n_samples)
        # thetas = np.linspace(eps, np.pi-eps, num=n_samples)
        phis = np.random.uniform(low=0, high=2*np.pi, size=n_samples)
        thetas = np.random.uniform(low=eps, high=np.pi-eps, size=n_samples)

        # actually calculate action
        action = Action(a, b, mu, mode=mode, cs=cs)

        phi0s = []
        phi2s = []
        Gs = []

        # xx, yy = np.meshgrid(phis, thetas)
        # coordinates = np.vstack([xx.ravel(), yy.ravel()]).T
        # np.random.shuffle(coordinates)

        coordinates = np.vstack([phis, thetas]).T

        for angles0 in tqdm(coordinates, total=n_samples):
            phi0, theta0 = angles0[0], angles0[1]
            phi0, phi2, G = action(phi0, theta0)

            #print(phi0*180/np.pi, phi2*180/np.pi, G)
            #action.returnmap.plot(phi0, theta0)

            if phi2 is not None and G is not None:
                # if i%1000 == 0:
                #    print(phi0*180/np.pi, phi2*180/np.pi)
                #    action.returnmap.plot(phi0, theta0)
                phi0s.append(phi0)
                phi2s.append(phi2)
                Gs.append(G)

        # phis = np.vstack((phi0s, phi2s)).T

        print(f"SAVING DATASET TO {filename}...")
        dataset = np.vstack((phi0s, phi2s, Gs)).T

        np.save(filename, dataset)

        # generate readme
        generate_readme(
            data_dir, f"a={a},\nb={b},\nmu={mu},\nn_samples={n_samples}")
