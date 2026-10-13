"""Multi-split replication: 5 splits x 3 seeds. Writes results/multisplit.csv"""
import warnings
import pandas as pd
from common import (load_adult, make_split, fit, scores, group_metrics,
                    matched_tpr_gap)

warnings.filterwarnings("ignore", category=UserWarning)

EPS_VALUES = [0.1, 0.5, 1.0, 8.0]
SPLITS = range(1, 6)
SEEDS = range(3)

X, y = load_adult()
rows = []
for sp in SPLITS:
    split = make_split(X, y, sp)
    for seed in SEEDS:
        base_model, _ = fit(split, seed)
        s_base = scores(base_model, split["Xte"])
        rate = (s_base > 0.5).mean()
        rows.append({"split": sp, "seed": seed, "eps": float("inf"),
                     **group_metrics(s_base, split),
                     "tpr_gap_matched": matched_tpr_gap(s_base, split, rate)})
        for eps in EPS_VALUES:
            model, info = fit(split, seed, eps=eps)
            s = scores(model, split["Xte"])
            rows.append({"split": sp, "seed": seed, "eps": eps, **info,
                         **group_metrics(s, split),
                         "tpr_gap_matched": matched_tpr_gap(s, split, rate)})
    print("split", sp, "done", flush=True)

pd.DataFrame(rows).to_csv("results/multisplit.csv", index=False)
print("wrote results/multisplit.csv with", len(rows), "rows (expect 75)")