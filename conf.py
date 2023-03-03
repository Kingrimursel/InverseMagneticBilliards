import os
import subprocess

from datetime import datetime
from io import BytesIO
import pandas as pd
import torch


def get_least_busy_gpu(verbose=True):
    gpu_stats = subprocess.check_output(
        ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(BytesIO(gpu_stats),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    gpu_df['memory.free'] = gpu_df['memory.free'].map(
        lambda x: x.rstrip(' [MiB]'))
    gpu_df["memory.free"] = pd.to_numeric(gpu_df["memory.free"])
    idx = gpu_df['memory.free'].idxmax()

    if verbose:
        print('GPU usage:\n{}'.format(gpu_df))
        print('Returning GPU{} with {} free MiB'.format(
            idx, gpu_df.iloc[idx]['memory.free']))

    return idx


# ńvidia-smi might not be availible
try:
    # determine least busy device
    least_busy_device = get_least_busy_gpu(verbose=True)
    device = torch.device(
        f"cuda:{least_busy_device}" if torch.cuda.is_available() else "cpu")
except FileNotFoundError:
    device = "cpu"



BASEDIR = os.path.dirname(__file__)
DATADIR = os.path.join(BASEDIR, "data/raw")
MODELDIR = os.path.join(BASEDIR, "output/models")
GRAPHICSDIR = os.path.join(BASEDIR, "output/graphics")

TODAY = datetime.today().strftime("%Y-%m-%d")

RES_TABLE = 1e3