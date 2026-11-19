"""ModelNet40 (official PointNet HDF5 release: 2048 points + normals per object, 9,840 train / 2,468 test)."""
import glob
import os
import zlib

import h5py
import numpy as np
from torch.utils.data import Dataset

import scan

DATA_DIR = os.environ.get("MODELNET_DIR",
                          os.path.expanduser("~/datasets/portfolio/modelnet40/modelnet40_ply_hdf5_2048"))
URL = "https://huggingface.co/datasets/Msun/modelnet40/resolve/main/modelnet40_ply_hdf5_2048.zip"


def load(split):
    files = sorted(glob.glob(os.path.join(DATA_DIR, f"ply_data_{split}*.h5")))
    if not files:
        raise FileNotFoundError(f"No ModelNet40 HDF5 in {DATA_DIR}; download {URL} and set MODELNET_DIR")
    P, N, L = [], [], []
    for f in files:
        with h5py.File(f, "r") as h:
            P.append(h["data"][:]); N.append(h["normal"][:]); L.append(h["label"][:, 0])
    return np.concatenate(P), np.concatenate(N), np.concatenate(L).astype(np.int64)


def class_names():
    with open(os.path.join(DATA_DIR, "shape_names.txt")) as f:
        return [l.strip() for l in f if l.strip()]


class Train(Dataset):
    """aug='cad': standard PointNet training (random 1024 of 2048 points, scale + jitter).
    aug='scan': random simulated depth scans (scan.random_scan) with the same scale + jitter."""

    def __init__(self, P, N, L, aug="cad"):
        self.P, self.N, self.L, self.aug = P, N, L, aug

    def __len__(self):
        return len(self.L)

    def __getitem__(self, i):
        rng = np.random.default_rng()
        if self.aug == "scan":
            x = scan.random_scan(self.P[i], self.N[i], rng)
        else:
            x = scan.normalise(self.P[i][rng.choice(2048, 1024, replace=False)])
        x = x * rng.uniform(0.8, 1.25, size=(1, 3)) + np.clip(0.01 * rng.standard_normal(x.shape), -0.05, 0.05)
        return x.astype(np.float32), self.L[i]


class Test(Dataset):
    """Deterministic test inputs: every object under `cond`, seen from `views` seeded random azimuths."""

    def __init__(self, P, N, L, cond, views=4):
        self.P, self.N, self.L, self.cond, self.views = P, N, L, cond, views
        self.kw = scan.condition(cond)

    def __len__(self):
        return len(self.L) * self.views

    def __getitem__(self, j):
        i, v = divmod(j, self.views)
        az = np.random.default_rng(zlib.crc32(f"{i}-{v}".encode())).uniform(0, 360)   # same views in every condition
        rng = np.random.default_rng(zlib.crc32(f"{i}-{v}-{self.cond}".encode()))
        if self.kw.get("view", True) is False:
            x = scan.normalise(self.P[i][rng.choice(2048, 1024, replace=False)]).astype(np.float32)
        else:
            x = scan.scan(self.P[i], self.N[i], rng, az=az, **self.kw)
        return x, self.L[i], i
