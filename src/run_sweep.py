"""Main sweep on the original split (random_state=0): TPR gap etc. vs epsilon.
Writes results/sweep_original_split.csv"""
import warnings
import pandas as pd
from common import load_adult, make_split, fit, scores, group_metrics

warnings.filterwarnings("ignore", category=UserWarning)

EPS_VALUES = [0.1, 0.5, 1.0, 4.0, 8.0]
SEEDS = range(10)

X, y = load_adult()
split = make_split(X, y, 0)

rows = []
for seed in SEEDS:
    model, _ = fit(split, seed)
    rows.append({"eps": float("inf"), "seed": seed,
                 **group_metrics(scores(model, split["Xte"]), split)})
    for eps in EPS_VALUES:
        model, info = fit(split, seed, eps=eps)
        rows.append({"eps": eps, "seed": seed, **info,
                     **group_metrics(scores(model, split["Xte"]), split)})
    print("seed", seed, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/sweep_original_split.csv", index=False)
print(df.groupby("eps")[["accuracy", "sel_gap", "tpr_gap", "fpr_gap"]]
        .agg(["mean", "std"]).round(3))
print("max |epsilon_spent - target| over DP runs:",
      (df.dropna(subset=["epsilon_spent"])["epsilon_spent"]
       - df.dropna(subset=["epsilon_spent"])["eps"]).abs().max().round(4))