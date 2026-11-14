"""Part 6 tuning extension: the best sgdm lr (0.03) sat at the lower edge of the
pre-registered grid, so the grid is extended downward as the protocol says."""
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from common import load_adult, make_split, fit, scores

warnings.filterwarnings("ignore", category=UserWarning)
EPS = 0.1
SEEDS = range(3)
NEW_LRS = [0.01, 0.003]

X, y = load_adult()
split0 = make_split(X, y, 0)
Xtr, ytr = split0["Xtr"], split0["ytr"]
idx = np.arange(len(ytr))
i_tr, i_va = train_test_split(idx, test_size=0.2, random_state=123,
                              stratify=ytr.numpy())
sub = {"Xtr": Xtr[i_tr], "ytr": ytr[i_tr]}
Xva, yva = Xtr[i_va], ytr[i_va].numpy()

rows = []
for lr in NEW_LRS:
    for seed in SEEDS:
        model, info = fit(sub, seed, eps=EPS, optimizer="sgdm", lr=lr)
        s = scores(model, Xva)
        bad = not np.isfinite(s).all()
        if bad:
            ll, auc = float("inf"), float("nan")
        else:
            s = np.clip(s, 1e-7, 1 - 1e-7)
            ll, auc = log_loss(yva, s), roc_auc_score(yva, s)
        rows.append({"arm": "sgdm", "lr": lr, "gamma": None, "seed": seed,
                     "val_logloss": ll, "val_auc": auc, "diverged": bad,
                     "noise_multiplier": info.get("noise_multiplier")})
    print("sgdm", lr, "done", flush=True)

new = pd.DataFrame(rows)
new.to_csv("results/tuning_part6_sgdm_ext.csv", index=False)
old = pd.read_csv("results/tuning_part6.csv")
old = old[old.arm == "sgdm"]
allr = pd.concat([old, new])
g = allr.groupby("lr").agg(val_logloss=("val_logloss", "mean"),
                           val_auc=("val_auc", "mean"),
                           diverged=("diverged", "sum")).reset_index()
print(g.to_string(index=False))
print("best sgdm lr:", g.sort_values("val_logloss").iloc[0]["lr"])
