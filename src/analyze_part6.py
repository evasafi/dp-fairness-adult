"""Part 6 analysis: change vs non-private baseline, arm contrasts."""
import numpy as np
import pandas as pd
from scipy import stats

df = pd.read_csv("results/part6.csv")
base = df[df.arm == "nonprivate"].set_index(["split", "seed"])
priv = df[df.arm != "nonprivate"].copy()
COLS = ["tpr_gap", "tpr_M", "tpr_F", "accuracy"]
for c in COLS:
    priv["d_" + c] = [r[c] - base.loc[(r["split"], r["seed"]), c]
                      for _, r in priv.iterrows()]
D = ["d_" + c for c in COLS]
per_split = priv.groupby(["arm", "split"])[D].mean().reset_index()

def ci(x, level):
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = x.mean()
    se = x.std(ddof=1) / np.sqrt(n)
    t = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, m - t * se, m + t * se

print("Per-arm change vs non-private at eps 0.1 (mean over 5 splits, 95% t-interval)")
for arm in ["A_adam", "B_adambc", "C_sgdm"]:
    sub = per_split[per_split.arm == arm]
    for d in D:
        m, lo, hi = ci(sub[d], 0.95)
        print(f"  {arm:9s} {d:11s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print()

print("Paired contrasts (97.5% t-interval across splits)")
for d in D:
    wide = per_split.pivot(index="split", columns="arm", values=d)
    for other in ["B_adambc", "C_sgdm"]:
        diff = wide[other] - wide["A_adam"]
        m, lo, hi = ci(diff, 0.975)
        print(f"  {other[0]} - A  {d:11s} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    print()

print("Per-split change in tpr_gap (mean over seeds)")
print(per_split.pivot(index="split", columns="arm", values="d_tpr_gap").round(4).to_string())
print()
print("Absolute mean tpr_gap by arm (mean over all 15 runs)")
print(df.groupby("arm").tpr_gap.mean().round(4).to_string())
print()
b = df[df.arm == "B_adambc"]
print("B floor fraction: mean %.3f, min %.3f, max %.3f" %
      (b.floor_frac.mean(), b.floor_frac.min(), b.floor_frac.max()))
