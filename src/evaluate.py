"""All experiments: every trained model on every test condition, then the analysis and figures.

    python src/evaluate.py      # -> cache/inputs/*.npy, cache/logits/*, results/results.json, results/figures/*.png

Test set: the 2,468 official ModelNet40 test objects, each seen from 4 seeded viewpoints (the same 4 azimuths in
every condition). Confidence intervals: 2.5-97.5 % bootstrap over objects (1,000 resamples).
"""
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

import data
import scan
from model import build

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE, RES = os.path.join(ROOT, "cache"), os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
VIEWS, BOOT = 4, 1000
MODELS = {"pointnet_cad": "PointNet", "dgcnn_cad": "DGCNN", "pointnet_scan": "PointNet + sim-scan",
          "dgcnn_scan": "DGCNN + sim-scan"}
COLORS = {"pointnet_cad": "#8da0cb", "dgcnn_cad": "#1f4e9c", "pointnet_scan": "#fc8d62", "dgcnn_scan": "#c0392b"}
FACTOR_LABEL = {"noise": "depth noise (camera distance)", "flying": "flying pixels", "holes": "holes",
                "plane": "table-plane leftovers"}
SEV_LABEL = {"noise": ["1 m", "1.5 m", "2 m"], "flying": ["5 %", "10 %", "20 %"], "holes": ["10 %", "25 %", "40 %"],
             "plane": ["10 %", "25 %", "40 %"]}


def make_inputs(P, N, L):
    os.makedirs(os.path.join(CACHE, "inputs"), exist_ok=True)
    for c in scan.ALL_CONDITIONS:
        f = os.path.join(CACHE, "inputs", f"{c}.npy")
        if os.path.exists(f):
            continue
        X = np.zeros((len(L) * VIEWS, 1024, 3), np.float16)
        dl = DataLoader(data.Test(P, N, L, c, VIEWS), batch_size=256, num_workers=20)
        k = 0
        for x, _, _ in dl:
            X[k:k + len(x)] = x.numpy(); k += len(x)
        np.save(f, X)
        print("inputs", c, flush=True)


@torch.no_grad()
def make_logits():
    dev = torch.device("cuda")
    for ck in sorted(glob.glob(os.path.join(CACHE, "*_s[0-9].pt"))):
        tag = os.path.basename(ck)[:-3]
        out = os.path.join(CACHE, "logits", f"{tag}.npz")
        if os.path.exists(out):
            continue
        m = build(tag.split("_")[0]).to(dev)
        m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()
        res = {}
        for c in scan.ALL_CONDITIONS:
            X = np.load(os.path.join(CACHE, "inputs", f"{c}.npy"), mmap_mode="r")
            outs = []
            for i in range(0, len(X), 32):
                with torch.autocast("cuda", dtype=torch.float16):
                    outs.append(m(torch.from_numpy(np.asarray(X[i:i + 32], np.float32)).to(dev)).float().cpu())
            res[c] = torch.cat(outs).numpy()
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out, **res)
        print("logits", tag, flush=True)


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def load_probs():
    """probs[model][cond] -> (n_obj, VIEWS, 40), averaged over training seeds (seeds listed in `seeds`)."""
    probs, seeds = {}, {}
    for m in MODELS:
        fs = sorted(glob.glob(os.path.join(CACHE, "logits", f"{m}_s*.npz")))
        if not fs:
            continue
        seeds[m] = len(fs)
        per_seed = [np.load(f) for f in fs]
        probs[m] = {c: [softmax(z[c]).reshape(-1, VIEWS, 40) for z in per_seed] for c in scan.ALL_CONDITIONS}
    return probs, seeds


def boot_ci(v, rng, stat=np.mean):
    """v: per-object values. 2.5/97.5 % bootstrap interval of `stat` over objects."""
    idx = rng.integers(0, len(v), size=(BOOT, len(v)))
    b = stat(v[idx], axis=1)
    return [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def acc_obj(plist, y):
    """Per-object accuracy averaged over views and seeds -> (n_obj,)."""
    return np.mean([(p.argmax(2) == y[:, None]).mean(1) for p in plist], 0)


def ece(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo in np.linspace(0, 1, bins + 1)[:-1]:
        k = (conf > lo) & (conf <= lo + 1 / bins)
        if k.any():
            e += k.mean() * abs((pred[k] == y[k]).mean() - conf[k].mean())
    return float(e)


def conformal(p_cal, y_cal, p_test, y_test, alpha=0.1):
    """LAC / 'score = 1 - p(true class)' split conformal prediction sets."""
    s = 1 - p_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = np.sort(s)[min(int(np.ceil((n + 1) * (1 - alpha))) - 1, n - 1)]
    sets = p_test >= 1 - q
    return float(sets[np.arange(len(y_test)), y_test].mean()), float(sets.sum(1).mean())


def main():
    os.makedirs(FIG, exist_ok=True)
    P, N, L = data.load("test")
    names = data.class_names()
    make_inputs(P, N, L)
    make_logits()
    probs, seeds = load_probs()
    rng = np.random.default_rng(0)
    R = {"n_objects": int(len(L)), "views": VIEWS, "seeds": seeds, "accuracy": {}, "calibration": {},
         "conformal": {}, "selective": {}, "per_class_drop": {}}

    # ---- accuracy everywhere ----
    for m in probs:
        R["accuracy"][m] = {}
        for c in scan.ALL_CONDITIONS:
            a = acc_obj(probs[m][c], L)
            R["accuracy"][m][c] = {"acc": float(a.mean()), "ci": boot_ci(a, rng)}
        a0 = acc_obj(probs[m]["cad"], L)
        for c in ["view", "realistic"]:
            d = a0 - acc_obj(probs[m][c], L)
            R["accuracy"][m][f"drop_cad_to_{c}"] = {"drop": float(d.mean()), "ci": boot_ci(d, rng)}
    # paired differences between models (same objects, same views)
    R["paired"] = {}
    pairs = [("dgcnn_cad", "pointnet_cad"), ("pointnet_scan", "pointnet_cad"), ("dgcnn_scan", "dgcnn_cad")]
    for a_, b_ in pairs:
        if a_ in probs and b_ in probs:
            R["paired"][f"{a_}-{b_}"] = {}
            for c in scan.ALL_CONDITIONS:
                d = acc_obj(probs[a_][c], L) - acc_obj(probs[b_][c], L)
                R["paired"][f"{a_}-{b_}"][c] = {"diff": float(d.mean()), "ci": boot_ci(d, rng)}

    # ---- does the model know when it is wrong? ----
    for m in probs:
        R["calibration"][m], R["selective"][m] = {}, {}
        for c in ["cad", "view", "realistic", "noise_s3", "plane_s3"]:
            p = probs[m][c][0].reshape(-1, 40); y = np.repeat(L, VIEWS)
            conf = p.max(1); cor = p.argmax(1) == y
            keep = conf >= np.quantile(conf, 0.2)
            R["calibration"][m][c] = {"ece": ece(p, y), "mean_conf": float(conf.mean()), "acc": float(cor.mean())}
            R["selective"][m][c] = {"acc_all": float(cor.mean()), "acc_top80": float(cor[keep].mean())}
    # conformal: calibrate on half of the objects (CAD or scans), test on the other half's scans; 200 splits
    for m in probs:
        R["conformal"][m] = {}
        for cal_c, test_c in [("cad", "cad"), ("cad", "realistic"), ("realistic", "realistic"), ("cad", "view")]:
            cov, size = [], []
            for r in range(200):
                perm = np.random.default_rng(r).permutation(len(L))
                ci, ti = perm[: len(L) // 2], perm[len(L) // 2:]
                pc, pt = probs[m][cal_c][0], probs[m][test_c][0]
                cv, sz = conformal(pc[ci].reshape(-1, 40), np.repeat(L[ci], VIEWS),
                                   pt[ti].reshape(-1, 40), np.repeat(L[ti], VIEWS))
                cov.append(cv); size.append(sz)
            R["conformal"][m][f"{cal_c}->{test_c}"] = {
                "coverage": float(np.mean(cov)), "coverage_ci": [float(np.percentile(cov, 2.5)),
                                                                  float(np.percentile(cov, 97.5))],
                "set_size": float(np.mean(size))}

    # ---- which classes collapse under single-view scans ----
    ref = "dgcnn_cad" if "dgcnn_cad" in probs else "pointnet_cad"
    a0, a1 = acc_obj(probs[ref]["cad"], L), acc_obj(probs[ref]["realistic"], L)
    per = sorted([(names[k], float(a0[L == k].mean()), float(a1[L == k].mean()), int((L == k).sum()))
                  for k in range(40)], key=lambda t: t[2] - t[1])
    R["per_class_drop"] = {"model": ref, "classes": per}
    pred = np.concatenate([p.argmax(2) for p in probs[ref]["realistic"]], 1)
    conf_pairs = {}
    for k in range(40):
        pk = pred[L == k].ravel(); pk = pk[pk != k]
        if len(pk):
            j = np.bincount(pk, minlength=40).argmax()
            conf_pairs[names[k]] = (names[j], float((pk == j).sum() / ((L == k).sum() * pred.shape[1])))
    R["per_class_drop"]["top_confusion_realistic"] = conf_pairs

    json.dump(R, open(os.path.join(RES, "results.json"), "w"), indent=1)
    figures(R, probs, P, N, L, names)
    summary(R)


def figures(R, probs, P, N, L, names):
    A = R["accuracy"]
    # 1. examples
    rng = np.random.default_rng(3)
    ex = [np.where(L == names.index(n))[0][0] for n in ["chair", "airplane", "lamp", "toilet"]]
    conds = ["cad", "view", "noise_s2", "flying_s2", "plane_s2", "realistic"]
    titles = ["CAD (benchmark)", "single view", "depth noise 1.5 m", "flying pixels 10 %", "table leftovers 25 %",
              "realistic scan"]
    fig = plt.figure(figsize=(3 * len(conds), 3 * len(ex)))
    for r, i in enumerate(ex):
        for c, (cn, tt) in enumerate(zip(conds, titles)):
            x = np.load(os.path.join(CACHE, "inputs", f"{cn}.npy"), mmap_mode="r")[i * VIEWS].astype(np.float32)
            ax = fig.add_subplot(len(ex), len(conds), r * len(conds) + c + 1, projection="3d")
            ax.scatter(x[:, 0], x[:, 2], x[:, 1], s=1.2, c=x[:, 1], cmap="viridis")
            ax.view_init(elev=20, azim=30); ax.set_axis_off()
            lim = 0.8; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
            if r == 0:
                ax.set_title(tt, fontsize=11)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "examples.png"), dpi=110); plt.close()

    # 2. severity curves
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.2), sharey=True)
    for ax, f in zip(axs, scan.FACTORS):
        for m in A:
            ys = [A[m]["view"]["acc"]] + [A[m][f"{f}_s{s}"]["acc"] for s in (1, 2, 3)]
            lo = [A[m]["view"]["ci"][0]] + [A[m][f"{f}_s{s}"]["ci"][0] for s in (1, 2, 3)]
            hi = [A[m]["view"]["ci"][1]] + [A[m][f"{f}_s{s}"]["ci"][1] for s in (1, 2, 3)]
            ax.plot(range(4), ys, "o-", color=COLORS[m], label=MODELS[m])
            ax.fill_between(range(4), lo, hi, color=COLORS[m], alpha=0.15)
        ax.set_xticks(range(4)); ax.set_xticklabels(["none"] + SEV_LABEL[f])
        ax.set_title(FACTOR_LABEL[f] + (" (not simulated in training)" if f == "plane" else ""), fontsize=10)
        ax.grid(alpha=0.3)
    axs[0].set_ylabel("accuracy (single-view scan + artefact)"); axs[0].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "severity.png"), dpi=130); plt.close()

    # 3. headline bars: CAD vs view vs realistic
    fig, ax = plt.subplots(figsize=(8, 4))
    cs = ["cad", "view", "realistic"]; w = 0.8 / len(A)
    for k, m in enumerate(A):
        v = [A[m][c]["acc"] for c in cs]
        e = np.array([[A[m][c]["acc"] - A[m][c]["ci"][0], A[m][c]["ci"][1] - A[m][c]["acc"]] for c in cs]).T
        ax.bar(np.arange(3) + (k - (len(A) - 1) / 2) * w, v, w, yerr=e, color=COLORS[m], label=MODELS[m], capsize=2)
    ax.set_xticks(range(3)); ax.set_xticklabels(["CAD surface (benchmark)", "single-view scan", "realistic scan"])
    ax.set_ylim(0, 1); ax.set_ylabel("test accuracy"); ax.legend(fontsize=8, ncol=2); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "headline.png"), dpi=130); plt.close()

    # 4. elevation
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for m in A:
        ax.plot([0, 30, 60, 90], [A[m][f"el{e}"]["acc"] for e in (0, 30, 60, 90)], "o-", color=COLORS[m],
                label=MODELS[m])
    ax.set_xlabel("camera elevation (deg), 90 = top-down"); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(FIG, "elevation.png"), dpi=130); plt.close()

    # 5. reliability: CAD vs realistic for the CAD-trained DGCNN / PointNet
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, m in zip(axs, [x for x in ["pointnet_cad", "dgcnn_cad"] if x in probs]):
        for c, col in [("cad", "#1f4e9c"), ("realistic", "#c0392b")]:
            p = probs[m][c][0].reshape(-1, 40); y = np.repeat(L, VIEWS)
            conf, cor = p.max(1), p.argmax(1) == y
            b = np.clip((conf * 10).astype(int), 0, 9)
            xs = [conf[b == i].mean() for i in range(10) if (b == i).sum() > 20]
            ys = [cor[b == i].mean() for i in range(10) if (b == i).sum() > 20]
            ax.plot(xs, ys, "o-", color=col, label=f"{c} (ECE {R['calibration'][m][c]['ece']:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_title(MODELS[m]); ax.set_xlabel("confidence")
        ax.set_ylabel("accuracy"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "reliability.png"), dpi=130); plt.close()

    # 6. per-class drop
    per = R["per_class_drop"]["classes"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(range(40), [t[1] for t in per], color="#bbbbbb", label="CAD")
    ax.bar(range(40), [t[2] for t in per], color="#c0392b", label="realistic scan")
    ax.set_xticks(range(40)); ax.set_xticklabels([t[0] for t in per], rotation=90, fontsize=8)
    ax.set_ylabel("accuracy"); ax.set_title(f"Per-class accuracy, {MODELS[R['per_class_drop']['model']]}")
    ax.legend(); plt.tight_layout(); plt.savefig(os.path.join(FIG, "per_class.png"), dpi=130); plt.close()


def summary(R):
    A = R["accuracy"]
    for m in A:
        print(f"{MODELS[m]:22s} " + " ".join(f"{c}={A[m][c]['acc']:.3f}" for c in scan.ALL_CONDITIONS))
    for m in R["conformal"]:
        print(m, {k: (round(v["coverage"], 3), round(v["set_size"], 2)) for k, v in R["conformal"][m].items()})
    for m in R["calibration"]:
        print(m, {k: round(v["ece"], 3) for k, v in R["calibration"][m].items()})


if __name__ == "__main__":
    main()
