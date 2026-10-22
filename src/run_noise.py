"""Part 3: noise vs clipping (P8-P10; predictions fixed in
notes/predictions.md). Writes results/noise_checks.csv"""
import warnings
import pandas as pd
from scipy import stats
from common import load_adult, make_split, fit, scores, group_metrics

warnings.filterwarnings("ignore", category=UserWarning)
EPS, SPLITS, SEEDS = 0.1, range(1, 6), range(3)
KS = [0.0, 0.25, 0.5, 1.0]
X, y = load_adult()

rows = []
for sp in SPLITS:
    split = make_split(X, y, sp)
    Xe = split["Xte"]
    for seed in SEEDS:
        m_np, _ = fit(split, seed)
        gap_np = group_metrics(scores(m_np, Xe), split)["tpr_gap"]
        # reference arm: the exact part 1 code path (target epsilon 0.1)
        m_ref, info = fit(split, seed, eps=EPS)
        sigma = float(info["noise_multiplier"])
        row = {"split": sp, "seed": seed, "sigma_star": sigma,
               "epsilon_spent": info["epsilon_spent"], "gap_np": gap_np,
               "chg_ref": group_metrics(scores(m_ref, Xe), split)["tpr_gap"] - gap_np}
        for k in KS:
            m_k, _ = fit(split, seed, noise_multiplier=k * sigma, clip=1.0)
            gap_k = group_metrics(scores(m_k, Xe), split)["tpr_gap"]
            row[f"chg_k{k}"] = gap_k - gap_np
        rows.append(row)
    print("split", sp, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/noise_checks.csv", index=False)


def across_splits(col):
    d = df.groupby("split")[col].mean()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d, tuple(round(float(x), 3) for x in ci)


print("\nsigma* (noise multiplier for epsilon 0.1):",
      round(df.sigma_star.min(), 3), round(df.sigma_star.max(), 3))
print("epsilon_spent range:", round(df.epsilon_spent.min(), 3),
      round(df.epsilon_spent.max(), 3))

d, ci = across_splits("chg_ref")
print("\nREFERENCE arm (part 1 code path; should be +0.067, CI 0.041 to 0.093):",
      round(d.mean(), 3), ci, d.round(3).tolist())

print("\nGap change (DP - non-private) by noise level k x sigma*")
means = {}
for k in KS:
    d, ci = across_splits(f"chg_k{k}")
    means[k] = d.mean()
    print(f"  k={k}: mean", round(d.mean(), 3), "CI", ci,
          "per split", d.round(3).tolist())

print("\nP10 (sanity, k=1.0 arm in part 1 CI 0.041 to 0.093):",
      "MET" if 0.041 <= means[1.0] <= 0.093 else
      "NOT MET (arms not comparable; do not interpret P8, P9)")

m0 = means[0.0]
print("P8 (k=0 gap change):", round(m0, 3),
      "MET" if m0 < 0.02 else ("NOT MET" if m0 > 0.04 else "INCONCLUSIVE"))

ordered = [means[k] for k in KS]
nondec = all(a <= b for a, b in zip(ordered, ordered[1:]))
big = means[1.0] - means[0.0] > 0.04
print("P9 (non-decreasing:", nondec, "; k=1 minus k=0 =",
      round(means[1.0] - means[0.0], 3), "):",
      "MET" if (nondec and big) else "NOT MET")