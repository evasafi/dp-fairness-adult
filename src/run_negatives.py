"""Part 2b: true-negative logit shifts (P4, P5; predictions fixed in
notes/predictions.md). Writes results/negatives_checks.csv"""
import warnings
import pandas as pd
import torch
from scipy import stats
from common import load_adult, make_split, fit

warnings.filterwarnings("ignore", category=UserWarning)
EPS, SPLITS, SEEDS = 0.1, range(1, 6), range(3)
X, y = load_adult()


def logits(model, Xe):
    model.eval()
    with torch.no_grad():
        return model(Xe).squeeze(1).numpy()


rows = []
for sp in SPLITS:
    split = make_split(X, y, sp)
    Xe, ye, male = split["Xte"], split["yte"], split["male"]
    groups = {"pos_M": male & (ye == 1), "pos_F": ~male & (ye == 1),
              "neg_M": male & (ye == 0), "neg_F": ~male & (ye == 0)}
    for seed in SEEDS:
        m_np, _ = fit(split, seed)
        m_dp, info = fit(split, seed, eps=EPS)
        l_np, l_dp = logits(m_np, Xe), logits(m_dp, Xe)
        row = {"split": sp, "seed": seed, "epsilon_spent": info["epsilon_spent"]}
        for name, mask in groups.items():
            row["shift_" + name] = l_dp[mask].mean() - l_np[mask].mean()
        rows.append(row)
    print("split", sp, "done", flush=True)

df = pd.DataFrame(rows)
df["pos_minus_neg_F"] = df.shift_pos_F - df.shift_neg_F
df.to_csv("results/negatives_checks.csv", index=False)


def across_splits(col):
    d = df.groupby("split")[col].mean()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d, tuple(round(float(x), 3) for x in ci)


print("\nSANITY (should match part 2a P1: men -0.048, women -0.665):",
      round(df.shift_pos_M.mean(), 3), round(df.shift_pos_F.mean(), 3))
print("epsilon_spent range:", round(df.epsilon_spent.min(), 3),
      round(df.epsilon_spent.max(), 3))

print("\nTrue-negative logit shift (DP - non-private)")
for col in ["shift_neg_M", "shift_neg_F"]:
    d, ci = across_splits(col)
    print(" ", col, "mean", round(d.mean(), 3), "CI", ci,
          "per split", d.round(3).tolist())

print("\nContext, not pre-specified: women's positive shift minus negative shift")
d, ci = across_splits("pos_minus_neg_F")
print("  mean", round(d.mean(), 3), "CI", ci)

d, ci = across_splits("shift_neg_F")
mean_f = d.mean()
if mean_f > -0.20:
    verdict = "P5 (positive-specific shift)"
elif -0.82 <= mean_f <= -0.52 and ci[1] < 0:
    verdict = "P4 (group-wide offset)"
elif mean_f < -0.82:
    verdict = "OUTSIDE both bands (more negative than -0.82)"
else:
    verdict = "MIXED (between -0.52 and -0.20)"
print("\nVerdict from women's true-negative shift:", verdict)