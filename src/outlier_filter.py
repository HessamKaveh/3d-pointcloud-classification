"""Does a standard pre-processing step rescue the CAD-trained models?

    python src/outlier_filter.py      # -> results/outlier_filter.json  (needs cache/inputs from evaluate.py)

Statistical outlier removal (mean distance to the 16 nearest neighbours > mean + 2 std is dropped),
then re-centre, re-scale to the unit sphere and re-sample back to 1024 points. This is what most
robot pipelines do before classification (e.g. PCL / Open3D `remove_statistical_outlier`).
"""
import glob
import json
import os

import numpy as np
import torch

import data
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache")
VIEWS = 4
CONDS = ["cad", "view", "flying_s2", "noise_s3", "plane_s2", "realistic"]


def sor(p, k=16, std=2.0, rng=None):
    p = p.astype(np.float32)
    d = torch.cdist(torch.from_numpy(p)[None], torch.from_numpy(p)[None])[0]
    md = d.topk(k + 1, largest=False).values[:, 1:].mean(1).numpy()
    keep = md <= md.mean() + std * md.std()
    q = p[keep]
    q = q - q.mean(0)
    q = q / (np.linalg.norm(q, axis=1).max() + 1e-9)
    idx = rng.choice(len(q), 1024, replace=len(q) < 1024)
    return q[idx], 1 - keep.mean()


@torch.no_grad()
def main():
    dev = torch.device("cuda")
    _, _, L = data.load("test")
    y = np.repeat(L, VIEWS)
    rng = np.random.default_rng(0)
    inputs, removed = {}, {}
    for c in CONDS:
        X = np.load(os.path.join(CACHE, "inputs", f"{c}.npy")).astype(np.float32)
        out = [sor(x, rng=rng) for x in X]
        inputs[c] = (X, np.stack([o[0] for o in out]))
        removed[c] = float(np.mean([o[1] for o in out]))
        print("filtered", c, f"removed {removed[c]:.3f}", flush=True)
    R = {"removed_fraction": removed, "acc": {}}
    for ck in sorted(glob.glob(os.path.join(CACHE, "*_s0.pt"))):
        tag = os.path.basename(ck)[:-3]
        m = build(tag.split("_")[0]).to(dev); m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()
        R["acc"][tag] = {}
        for c in CONDS:
            accs = []
            for X in inputs[c]:
                pred = []
                for i in range(0, len(X), 32):
                    with torch.autocast("cuda", dtype=torch.float16):
                        pred.append(m(torch.from_numpy(X[i:i + 32]).to(dev)).argmax(1).cpu())
                accs.append(float((torch.cat(pred).numpy() == y).mean()))
            R["acc"][tag][c] = {"raw": accs[0], "filtered": accs[1]}
            print(f"{tag:18s} {c:10s} raw {accs[0]:.3f} filtered {accs[1]:.3f}", flush=True)
    json.dump(R, open(os.path.join(ROOT, "results", "outlier_filter.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
