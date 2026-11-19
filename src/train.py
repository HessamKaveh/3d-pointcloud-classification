"""Train a point-cloud classifier on ModelNet40.

    python src/train.py --model pointnet --aug cad  --seed 0     # standard benchmark training
    python src/train.py --model dgcnn    --aug scan --seed 0     # trained on simulated depth scans

A fixed epoch budget is used and the last epoch is kept (no checkpoint selection on the test set).
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import data
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {"pointnet": dict(epochs=200, batch=32), "dgcnn": dict(epochs=100, batch=16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="pointnet", choices=["pointnet", "dgcnn"])
    ap.add_argument("--aug", default="cad", choices=["cad", "scan"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    cfg = DEFAULTS[a.model]
    epochs = a.epochs or cfg["epochs"]
    tag = f"{a.model}_{a.aug}_s{a.seed}"
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda")

    P, N, L = data.load("train")
    loader = DataLoader(data.Train(P, N, L, a.aug), batch_size=cfg["batch"], shuffle=True, drop_last=True,
                        num_workers=a.workers, persistent_workers=True,
                        worker_init_fn=lambda w: np.random.seed((a.seed * 1000 + w) % 2**32))
    Pt, Nt, Lt = data.load("test")
    test = DataLoader(data.Test(Pt, Nt, Lt, "cad", views=1), batch_size=64, num_workers=8)

    model = build(a.model).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(loader), pct_start=0.05)
    crit = nn.CrossEntropyLoss(label_smoothing=0.2)
    scaler = torch.amp.GradScaler()
    hist = {"loss": [], "train_acc": [], "cad_acc": [], "sec": []}
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        tot = cor = n = 0
        for x, y in loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x)
                loss = crit(out.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y); cor += (out.argmax(1) == y).sum().item(); n += len(y)
        hist["loss"].append(tot / n); hist["train_acc"].append(cor / n); hist["sec"].append(time.time() - t0)
        if ep % 10 == 9 or ep == epochs - 1:     # monitoring only; never used to pick a checkpoint
            model.eval(); c = m = 0
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for x, y, _ in test:
                    c += (model(x.to(dev)).argmax(1).cpu() == y).sum().item(); m += len(y)
            hist["cad_acc"].append((ep + 1, c / m))
            print(f"{tag} ep {ep + 1}/{epochs} loss {tot / n:.3f} train {cor / n:.3f} cad-test {c / m:.3f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    os.makedirs(os.path.join(ROOT, "cache"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "results", "history"), exist_ok=True)
    torch.save(model.state_dict(), os.path.join(ROOT, "cache", f"{tag}.pt"))
    json.dump(hist, open(os.path.join(ROOT, "results", "history", f"{tag}.json"), "w"))


if __name__ == "__main__":
    main()
