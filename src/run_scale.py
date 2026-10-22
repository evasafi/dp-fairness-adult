"""Part 2c: logit scale check (P6, P7; predictions fixed in
notes/predictions.md). Writes results/scale_checks.csv"""
import warnings
import numpy as np
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


def dprime(l, pos, neg):
    pooled = np.sqrt((l[pos].var(ddof=1) + l[neg].var(ddof=1)) / 2)
    return (l[pos].mean() - l[neg].mean()) / pooled


rows = []
for sp in SPLITS:
    split = make_split(X, y, sp)
    Xe, ye, male = split["Xte"], split["yte"], split["male"]
    cells = {"pos_M": male & (ye == 1), "pos_F": ~male & (ye == 1),
             "neg_M": male & (ye == 0), "neg_F": ~male & (ye == 0)}
    for seed in SEEDS:
        m_np, _ = fit(split, seed)
        m_dp, info = fit(split, seed, eps=EPS)
        l_np, l_dp = logits(m_np, Xe), logits(m_dp, Xe)
        row = {"split": sp, "seed": seed, "epsilon_spent": info["epsilon_spent"],
               "shift_pos_M": l_dp[cells["pos_M"]].mean() - l_np[cells["pos_M"]].mean(),
               "shift_pos_F": l_dp[cells["pos_F"]].mean() - l_np[cells["pos_F"]].mean()}
        for name, mask in cells.items():
            row["sd_np_" + name] = l_np[mask].std(ddof=1)
            row["sd_dp_" + name] = l_dp[mask].std(ddof=1)
            row["ratio_" + name] = row["sd_dp_" + name] / row["sd_np_" + name]
        for g, sex in (("M", male), ("F", ~male)):
            pos, neg = cells["pos_" + g], cells["neg_" + g]
            row["dprime_np_" + g] = dprime(l_np, pos, neg)
            row["dprime_dp_" + g] = dprime(l_dp, pos, neg)
            row["dprime_change_" + g] = row["dprime_dp_" + g] - row["dprime_np_" + g]
        rows.append(row)
    print("split", sp, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/scale_checks.csv", index=False)


def across_splits(col):
    d = df.groupby("split")[col].mean()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d, tuple(round(float(x), 3) for x in ci)


print("\nSANITY (should match part 2a P1: men -0.048, women -0.665):",
      round(df.shift_pos_M.mean(), 3), round(df.shift_pos_F.mean(), 3))
print("epsilon_spent range:", round(df.epsilon_spent.min(), 3),
      round(df.epsilon_spent.max(), 3))

print("\nP6: SD ratio (DP / non-private) by cell")
ratios = {}
for cell in ["pos_M", "pos_F", "neg_M", "neg_F"]:
    d, ci = across_splits("ratio_" + cell)
    ratios[cell] = (d.mean(), ci)
    print(" ", cell, "mean", round(d.mean(), 3), "CI", ci,
          "per split", d.round(3).tolist())
means = [v[0] for v in ratios.values()]
if all(v[0] > 1.2 and v[1][0] > 1 for v in ratios.values()):
    v6 = "MET (scale expansion)"
elif sum(0.8 <= m <= 1.2 for m in means) >= 3:
    v6 = "NOT MET: scale about unchanged"
elif sum(m < 0.8 for m in means) >= 3:
    v6 = "NOT MET: compression"
else:
    v6 = "NOT MET: mixed"
print("  P6:", v6)

print("\nP7: d' (separation in pooled-SD units), DP minus non-private")
dm, cim = across_splits("dprime_change_M")
dfm, cif = across_splits("dprime_change_F")
print("  men  : mean", round(dm.mean(), 3), "CI", cim,
      "per split", dm.round(3).tolist())
print("  women: mean", round(dfm.mean(), 3), "CI", cif,
      "per split", dfm.round(3).tolist())
diff = abs(dm.mean() - dfm.mean())
print("  |men - women| difference in d' change:", round(float(diff), 3))
falls = bool((dm < 0).all() and (dfm < 0).all())
print("  P7:", "MET" if (falls and diff < 0.1) else "NOT MET",
      "(falls in all splits for both groups:", falls, ")")