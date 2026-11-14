"""Part 6 final run: DP-Adam (A), DP-AdamBC (B), DP-SGD momentum (C) at eps 0.1.
Hyperparameters fixed in notes/predictions.md before this script was run.
Writes results/part6.csv (60 rows: 5 splits x 3 seeds x [non-private + 3 arms])."""
import warnings
import numpy as np
import pandas as pd
import adambc
from common import (load_adult, make_split, fit, scores, group_metrics,
                    matched_tpr_gap)

warnings.filterwarnings("ignore", category=UserWarning)

# Diagnostic hook: remember the most recently built AdamBC optimizer.
_orig_init = adambc.AdamBC.__init__
def _init(self, *a, **k):
    _orig_init(self, *a, **k)
    adambc.LAST = self
adambc.AdamBC.__init__ = _init

EPS = 0.1
SPLITS = range(1, 6)
SEEDS = range(3)
ARMS = {
    "A_adam": dict(optimizer="adam", lr=0.01),
    "B_adambc": dict(optimizer="adambc", lr=0.003, gamma=1e-4),
    "C_sgdm": dict(optimizer="sgdm", lr=0.03),
}

X, y = load_adult()
rows = []
for sp in SPLITS:
    split = make_split(X, y, sp)
    for seed in SEEDS:
        base_model, _ = fit(split, seed)
        s_base = scores(base_model, split["Xte"])
        rate = (s_base > 0.5).mean()
        rows.append({"split": sp, "seed": seed, "arm": "nonprivate",
                     "eps": float("inf"), **group_metrics(s_base, split),
                     "tpr_gap_matched": matched_tpr_gap(s_base, split, rate)})
        for arm, kw in ARMS.items():
            model, info = fit(split, seed, eps=EPS, **kw)
            s = scores(model, split["Xte"])
            row = {"split": sp, "seed": seed, "arm": arm, "eps": EPS, **info,
                   **group_metrics(s, split),
                   "tpr_gap_matched": matched_tpr_gap(s, split, rate)}
            if arm == "B_adambc":
                o = adambc.LAST
                row["floor_frac"] = 1.0 - o.n_above / max(o.n_total, 1)
            rows.append(row)
    print("split", sp, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/part6.csv", index=False)
print("wrote results/part6.csv with", len(df), "rows (expect 60)")

# Cross-check: arm A must reproduce stored part 1 rows at eps 0.1.
old = pd.read_csv("results/multisplit.csv")
old = old[old.eps == 0.1][["split", "seed", "tpr_gap", "accuracy"]]
a = df[df.arm == "A_adam"][["split", "seed", "tpr_gap", "accuracy"]]
m = a.merge(old, on=["split", "seed"], suffixes=("_new", "_old"))
print("arm A vs stored part 1, max abs diff tpr_gap:",
      float(np.abs(m.tpr_gap_new - m.tpr_gap_old).max()),
      " accuracy:", float(np.abs(m.accuracy_new - m.accuracy_old).max()))
print("floor fraction (B), mean over runs:",
      float(df[df.arm == "B_adambc"].floor_frac.mean()))
