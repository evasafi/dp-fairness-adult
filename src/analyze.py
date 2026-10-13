"""Compute all reported statistics and figures from results/*.csv"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from scipy import stats

INF = float("inf")
ms = pd.read_csv("results/multisplit.csv")
sw = pd.read_csv("results/sweep_original_split.csv")

# ---- multi-split: paired change vs non-private within each (split, seed) ----
base = ms[ms.eps == INF].set_index(["split", "seed"])
EPS = [0.1, 0.5, 1.0, 8.0]
print("Non-private TPR gap by split:",
      base["tpr_gap"].groupby("split").mean().round(3).tolist(),
      "| mean", round(base["tpr_gap"].groupby("split").mean().mean(), 3),
      "| sd", round(base["tpr_gap"].groupby("split").mean().std(), 3))

def paired(col, eps):
    d = (ms[ms.eps == eps].set_index(["split", "seed"])[col] - base[col]).groupby("split").mean()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d, ci

print("\nMULTI-SPLIT: paired change vs non-private (mean over seeds within split; CI across 5 splits)")
for col in ["tpr_gap", "tpr_gap_matched", "tpr_M", "tpr_F", "accuracy", "auc_M", "auc_F"]:
    for eps in EPS:
        d, ci = paired(col, eps)
        print(f"{col:16s} eps={eps:<4} mean {d.mean():+.3f}  CI ({ci[0]:+.3f}, {ci[1]:+.3f})  "
              f"per split {d.round(3).tolist()}")
    print()

# ---- figure: primary multi-split figure ----
chg = {e: paired("tpr_gap", e)[0] for e in EPS}
fig, ax = plt.subplots(figsize=(6, 4))
for s in range(1, 6):
    ax.plot(EPS, [chg[e][s] for e in EPS], color="gray", alpha=0.5, lw=1, marker="o", ms=3,
            label="individual split" if s == 1 else None)
means = [chg[e].mean() for e in EPS]
cis = [chg[e].sem() * stats.t.ppf(0.975, 4) for e in EPS]
ax.errorbar(EPS, means, yerr=cis, fmt="o-", color="C0", capsize=3, lw=2,
            label="mean across splits (95% CI)")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_xscale("log")
ax.set_xticks(EPS)
ax.set_xticklabels(["0.1", "0.5", "1", "8"])
ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("epsilon (smaller = more private)")
ax.set_ylabel("change in TPR gap vs non-private")
ax.set_title("Change in TPR gap vs privacy budget (5 splits x 3 seeds)", fontsize=9)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/tpr_gap_change_multisplit.png", dpi=200)
plt.close()

# ---- original split (exploratory) ----
g = sw.groupby("eps")
print("ORIGINAL SPLIT (exploratory), mean over 10 seeds:")
print(g[["accuracy", "tpr_M", "tpr_F", "tpr_gap"]].mean().round(3))
b0 = sw[sw.eps == INF].set_index("seed")["tpr_gap"]
for eps in [0.1, 0.5, 1.0, 4.0, 8.0]:
    d = sw[sw.eps == eps].set_index("seed")["tpr_gap"] - b0
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    print(f"eps={eps:<4} change in TPR gap {d.mean():+.3f}  CI ({ci[0]:+.3f}, {ci[1]:+.3f})")
dp = sw.dropna(subset=["epsilon_spent"])
print("epsilon spent / target: min %.3f, max %.3f (ratio)" %
      ((dp.epsilon_spent / dp.eps).min(), (dp.epsilon_spent / dp.eps).max()))
print("\nwrote figures/tpr_gap_change_multisplit.png")
# ---- clip experiments (original split) ----
cl = pd.read_csv("results/clip_experiments.csv")
t = cl.groupby(["kind", "clip"], dropna=False).agg(
    gap=("tpr_gap_matched", "mean"), gap_sd=("tpr_gap_matched", "std"),
    scoreM=("mean_score_M", "mean"), scoreF=("mean_score_F", "mean"),
    aucM=("auc_M", "mean"), aucF=("auc_F", "mean")).round(3)
print("\nCLIP EXPERIMENTS (original split, 10 seeds; matched-selection TPR gap)")
print(t.to_string())