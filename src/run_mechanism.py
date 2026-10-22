"""Mechanism checks P1-P3 (predictions fixed in notes/predictions.md).
Writes results/mechanism_checks.csv"""
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
from common import load_adult, build_preprocessor, make_split, fit, scores

warnings.filterwarnings("ignore", category=UserWarning)
EPS, SPLITS, SEEDS = 0.1, range(1, 6), range(3)
X, y = load_adult()


def logits(model, Xe):
    model.eval()
    with torch.no_grad():
        return model(Xe).squeeze(1).numpy()


def sex_cols(split_seed, n_features):
    Xtr, _, _, _ = train_test_split(X, y, test_size=0.3, random_state=split_seed, stratify=y)
    names = list(build_preprocessor(X).fit(Xtr).get_feature_names_out())
    assert len(names) == n_features, (len(names), n_features)
    i_m = [i for i, n in enumerate(names) if n.endswith("sex_Male")]
    i_f = [i for i, n in enumerate(names) if n.endswith("sex_Female")]
    assert len(i_m) == 1 and len(i_f) == 1, (i_m, i_f)
    return i_m[0], i_f[0]


def sex_diff(model, i_m, i_f):
    w = next(model.parameters()).detach().numpy().reshape(-1)
    return float(w[i_m] - w[i_f])


def tpr(p, m, ye):
    return p[m & (ye == 1)].mean()


def gaps(s_np, s_dp, split):
    ye, male = split["yte"], split["male"]
    p_np, p_dp = s_np > 0.5, s_dp > 0.5
    pm = np.zeros(len(ye), dtype=bool)
    for m in (male, ~male):
        rate = p_np[m].mean()
        thr = np.quantile(s_dp[m], 1 - rate)
        pm[m] = s_dp[m] > thr
    return {"gap_np": tpr(p_np, male, ye) - tpr(p_np, ~male, ye),
            "gap_dp_t05": tpr(p_dp, male, ye) - tpr(p_dp, ~male, ye),
            "gap_dp_groupmatched": tpr(pm, male, ye) - tpr(pm, ~male, ye)}


rows = []
for sp in SPLITS:
    split = make_split(X, y, sp)
    i_m, i_f = sex_cols(sp, split["Xtr"].shape[1])
    Xe, ye, male = split["Xte"], split["yte"], split["male"]
    pos_m, pos_f = male & (ye == 1), ~male & (ye == 1)
    for seed in SEEDS:
        m_np, _ = fit(split, seed)
        m_dp, info = fit(split, seed, eps=EPS)
        l_np, l_dp = logits(m_np, Xe), logits(m_dp, Xe)
        s_np, s_dp = scores(m_np, Xe), scores(m_dp, Xe)
        rows.append({"split": sp, "seed": seed,
                     "shift_pos_M": l_dp[pos_m].mean() - l_np[pos_m].mean(),
                     "shift_pos_F": l_dp[pos_f].mean() - l_np[pos_f].mean(),
                     "sexdiff_np": sex_diff(m_np, i_m, i_f),
                     "sexdiff_dp": sex_diff(m_dp, i_m, i_f),
                     "epsilon_spent": info["epsilon_spent"], **gaps(s_np, s_dp, split)})
    print("split", sp, "done", flush=True)

df = pd.DataFrame(rows)
df["p1_diff"] = df.shift_pos_F - df.shift_pos_M
df["p3_change"] = df.sexdiff_dp - df.sexdiff_np
df["gap_change_t05"] = df.gap_dp_t05 - df.gap_np
df["gap_change_groupmatched"] = df.gap_dp_groupmatched - df.gap_np
df.to_csv("results/mechanism_checks.csv", index=False)


def across_splits(col):
    d = df.groupby("split")[col].mean()
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return d, ci


d, ci = across_splits("gap_change_t05")
print("\nSANITY (should match part 1: +0.067, CI 0.041 to 0.093):",
      round(d.mean(), 3), tuple(round(float(x), 3) for x in ci), d.round(3).tolist())
print("epsilon_spent range:", round(df.epsilon_spent.min(), 3), round(df.epsilon_spent.max(), 3))

print("\nP1: mean logit shift among true positives (DP - non-private)")
print("  men:", round(df.shift_pos_M.mean(), 3), " women:", round(df.shift_pos_F.mean(), 3))
d, ci = across_splits("p1_diff")
met = bool((d < 0).all() and ci[1] < 0)
print("  (women - men) per split:", d.round(3).tolist(),
      " mean", round(d.mean(), 3), " CI", tuple(round(float(x), 3) for x in ci))
print("  P1:", "MET" if met else "NOT MET")

print("\nP2: recall-gap change with group-specific thresholds")
d, ci = across_splits("gap_change_groupmatched")
m2 = d.mean()
v2 = "MET" if m2 < 0.02 else ("NOT MET" if m2 > 0.04 else "INCONCLUSIVE")
print("  per split:", d.round(3).tolist(), " mean", round(m2, 3),
      " CI", tuple(round(float(x), 3) for x in ci))
print("  P2:", v2)

print("\nP3: change in (w_Male - w_Female), DP - non-private")
ch = df.p3_change
spread = float(np.sqrt(df.groupby("split").p3_change.var(ddof=1).mean()))
print("  non-private mean sexdiff:", round(df.sexdiff_np.mean(), 3),
      " DP mean sexdiff:", round(df.sexdiff_dp.mean(), 3))
print("  change: mean", round(ch.mean(), 3), " positive in", int((ch > 0).sum()), "of", len(ch),
      " pooled within-split sd", round(spread, 3))
print("  P3:", "MET" if ((ch > 0).all() and ch.mean() > spread) else "NOT MET")