"""Gradient-norm diagnostic: which (group, label) cells get clipped in DP-SGD?
Linear model + BCE: per-example gradient norm = |p - y| * sqrt(||x||^2 + 1).
Writes results/gradient_diagnostic.csv"""
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from common import load_adult, make_split, fit

warnings.filterwarnings("ignore", category=UserWarning)

X, y = load_adult()
split = make_split(X, y, 0)
Xtr_df, _, ytr_s, _ = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
male_tr = (Xtr_df["sex"] == "Male").values
assert len(male_tr) == split["Xtr"].shape[0]
assert (ytr_s.values == split["ytr"].numpy()).all()

Xt, yt = split["Xtr"], split["ytr"]
xnorm = torch.sqrt((Xt ** 2).sum(1) + 1.0).numpy()
ylab = yt.numpy().astype(int)

CELLS = [("men", 1, True), ("men", 0, True), ("women", 1, False), ("women", 0, False)]


def per_example_norms(model):
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xt).squeeze(1)).numpy()
    return np.abs(p - ylab) * xnorm


rows = []
for seed in [0, 1]:
    for epochs in [0, 1, 10]:
        model, _ = fit(split, seed, epochs=epochs)
        g = per_example_norms(model)
        for grp, label, is_male in CELLS:
            m = (male_tr == is_male) & (ylab == label)
            rows.append({"seed": seed, "epochs": epochs, "group": grp, "label": label,
                         "n": int(m.sum()), "mean_norm": g[m].mean(),
                         "median_norm": np.median(g[m]),
                         "mean_xnorm": xnorm[m].mean(),
                         "frac_gt_0.1": (g[m] > 0.1).mean(),
                         "frac_gt_1": (g[m] > 1.0).mean(),
                         "frac_gt_5": (g[m] > 5.0).mean()})

df = pd.DataFrame(rows)
df.to_csv("results/gradient_diagnostic.csv", index=False)
cols = ["n", "mean_norm", "median_norm", "mean_xnorm", "frac_gt_0.1", "frac_gt_1", "frac_gt_5"]
print(df.groupby(["epochs", "group", "label"])[cols].mean().round(3).to_string())
spread = df.groupby(["epochs", "group", "label"])["mean_norm"].agg(lambda s: s.max() - s.min())
print("max difference in mean_norm across the 2 seeds:", round(float(spread.max()), 4))