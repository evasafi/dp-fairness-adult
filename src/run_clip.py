"""Original split: clipping sensitivity at eps=0.1 and clip-only (no noise) ablation.
Writes results/clip_experiments.csv"""
import warnings
import pandas as pd
from common import (load_adult, make_split, fit, scores, group_metrics,
                    matched_tpr_gap)

warnings.filterwarnings("ignore", category=UserWarning)
X, y = load_adult()
split = make_split(X, y, 0)
male = split["male"]
rows = []

def record(kind, clip, seed, rate, model):
    s = scores(model, split["Xte"])
    m = group_metrics(s, split)
    rows.append({"kind": kind, "clip": clip, "seed": seed,
                 "mean_score_M": s[male].mean(), "mean_score_F": s[~male].mean(),
                 "tpr_gap_matched": matched_tpr_gap(s, split, rate),
                 "auc_M": m["auc_M"], "auc_F": m["auc_F"]})

for seed in range(10):
    base_model, _ = fit(split, seed)
    rate = (scores(base_model, split["Xte"]) > 0.5).mean()
    record("baseline", float("nan"), seed, rate, base_model)
    for clip in [0.1, 1.0, 5.0]:
        model, _ = fit(split, seed, eps=0.1, clip=clip)
        record("dp_eps0.1", clip, seed, rate, model)
    for clip in [1e6, 5.0, 1.0, 0.1]:
        model, _ = fit(split, seed, noise_multiplier=0.0, clip=clip)
        record("clip_only", clip, seed, rate, model)
    print("seed", seed, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/clip_experiments.csv", index=False)
print(df.groupby(["kind", "clip"], dropna=False)[
    ["mean_score_M", "mean_score_F", "tpr_gap_matched", "auc_M", "auc_F"]
].agg(["mean", "std"]).round(3).to_string())