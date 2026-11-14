"""Part 6 tuning. Learning rate (and gamma for AdamBC) per arm, chosen by validation
log-loss on a holdout from split 0's training part, never by the gap.
Writes results/tuning_part6.csv"""
import itertools
import time
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from common import load_adult, make_split, fit, scores

warnings.filterwarnings("ignore", category=UserWarning)

EPS = 0.1
SEEDS = range(3)
GRIDS = {
    "adam": {"lr": [0.001, 0.003, 0.01, 0.03, 0.1], "gamma": [None]},
    "adambc": {"lr": [0.001, 0.003, 0.01, 0.03, 0.1],
               "gamma": [1e-8, 1e-6, 1e-4, 1e-3]},
    "sgdm": {"lr": [0.03, 0.1, 0.3, 1.0, 3.0], "gamma": [None]},
}

X, y = load_adult()
split0 = make_split(X, y, 0)
Xtr, ytr = split0["Xtr"], split0["ytr"]
idx = np.arange(len(ytr))
i_tr, i_va = train_test_split(idx, test_size=0.2, random_state=123,
                              stratify=ytr.numpy())
sub = {"Xtr": Xtr[i_tr], "ytr": ytr[i_tr]}
Xva, yva = Xtr[i_va], ytr[i_va].numpy()

rows = []
t0 = time.time()
for arm, grid in GRIDS.items():
    for lr, gamma in itertools.product(grid["lr"], grid["gamma"]):
        for seed in SEEDS:
            kw = {"optimizer": arm, "lr": lr}
            if gamma is not None:
                kw["gamma"] = gamma
            model, info = fit(sub, seed, eps=EPS, **kw)
            s = scores(model, Xva)
            bad = not np.isfinite(s).all()
            if bad:
                ll, auc = float("inf"), float("nan")
            else:
                s = np.clip(s, 1e-7, 1 - 1e-7)
                ll, auc = log_loss(yva, s), roc_auc_score(yva, s)
            rows.append({"arm": arm, "lr": lr, "gamma": gamma, "seed": seed,
                         "val_logloss": ll, "val_auc": auc, "diverged": bad,
                         "noise_multiplier": info.get("noise_multiplier")})
        print(arm, lr, gamma, "done", round(time.time() - t0), "s", flush=True)
    pd.DataFrame(rows).to_csv("results/tuning_part6.csv", index=False)

df = pd.DataFrame(rows)
g = df.groupby(["arm", "lr", "gamma"], dropna=False).agg(
    val_logloss=("val_logloss", "mean"), val_auc=("val_auc", "mean"),
    diverged=("diverged", "sum")).reset_index()
print(g.to_string(index=False))
for arm in GRIDS:
    best = g[g.arm == arm].sort_values("val_logloss").iloc[0]
    print("best", arm, dict(best))
