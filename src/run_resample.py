"""Part 5: controlled resampling in Adult (P14-P17; predictions fixed in
notes/predictions.md). Writes results/resample_checks.csv and
results/resample_output.txt"""
import warnings
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.model_selection import train_test_split
from common import load_adult, make_split, fit, scores, group_metrics

warnings.filterwarnings("ignore", category=UserWarning)
EPS, N, SPLITS, SEEDS = 0.1, 12000, range(1, 6), range(3)
# (name, women's share, women's rate, men's rate); None = natural value
ARMS = [("Ref", None, None, None), ("A1", 0.15, None, None),
        ("A2", 0.50, None, None), ("B1", None, 0.20, None),
        ("B2", None, 0.25, None), ("B3", None, None, "equal")]
X, y = load_adult()
lines = []


def say(*a):
    s = " ".join(str(v) for v in a)
    print(s)
    lines.append(s)


def train_cells(sp, split):
    Xtr, _, ytr, _ = train_test_split(
        X, y, test_size=0.3, random_state=sp, stratify=y)
    assert np.array_equal(split["ytr"].numpy(), ytr.values), "row order mismatch"
    fem = (Xtr["sex"] == "Female").values
    pos = ytr.values == 1
    return {"FP": np.where(fem & pos)[0], "FN": np.where(fem & ~pos)[0],
            "MP": np.where(~fem & pos)[0], "MN": np.where(~fem & ~pos)[0]}


def plan(cells, share, rf, rm):
    nf = len(cells["FP"]) + len(cells["FN"])
    nm = len(cells["MP"]) + len(cells["MN"])
    nat_rf, nat_rm = len(cells["FP"]) / nf, len(cells["MP"]) / nm
    s = nf / (nf + nm) if share is None else share
    f = nat_rf if rf is None else rf
    m = nat_rm if rm is None else (nat_rf if rm == "equal" else rm)
    n_f = int(round(s * N))
    n_m = N - n_f
    n_fp, n_mp = int(round(f * n_f)), int(round(m * n_m))
    return {"FP": n_fp, "FN": n_f - n_fp, "MP": n_mp, "MN": n_m - n_mp}


splits, cells_by = {}, {}
for sp in SPLITS:
    splits[sp] = make_split(X, y, sp)
    cells_by[sp] = train_cells(sp, splits[sp])

bad = []
for sp in SPLITS:
    for name, share, rf, rm in ARMS:
        for c, n in plan(cells_by[sp], share, rf, rm).items():
            if n < 0 or n > len(cells_by[sp][c]):
                bad.append((sp, name, c, n, len(cells_by[sp][c])))
if bad:
    raise SystemExit(f"Cells not available (split, arm, cell, need, have): {bad}")
say("Availability check passed for all 5 splits and 6 arms.")

say("\nTraining composition, split 1:")
for name, share, rf, rm in ARMS:
    p = plan(cells_by[1], share, rf, rm)
    nf, nm = p["FP"] + p["FN"], p["MP"] + p["MN"]
    say(" ", name, p, "| women's share", round(nf / N, 3),
        "| women's rate", round(p["FP"] / nf, 3),
        "| men's rate", round(p["MP"] / nm, 3))

rows = []
for sp in SPLITS:
    split, cells = splits[sp], cells_by[sp]
    Xe = split["Xte"]
    for ai, (name, share, rf, rm) in enumerate(ARMS):
        need = plan(cells, share, rf, rm)
        rng = np.random.default_rng(10_000 * sp + ai)
        idx = np.concatenate([rng.choice(cells[c], size=need[c], replace=False)
                              for c in ("FP", "FN", "MP", "MN")])
        idx_t = torch.as_tensor(idx, dtype=torch.long)
        sub = dict(split)
        sub["Xtr"], sub["ytr"] = split["Xtr"][idx_t], split["ytr"][idx_t]
        for seed in SEEDS:
            m_np, _ = fit(sub, seed)
            m_dp, info = fit(sub, seed, eps=EPS)
            gap_np = group_metrics(scores(m_np, Xe), split)["tpr_gap"]
            gap_dp = group_metrics(scores(m_dp, Xe), split)["tpr_gap"]
            rows.append({"split": sp, "arm": name, "seed": seed,
                         "gap_np": gap_np, "gap_dp": gap_dp,
                         "change": gap_dp - gap_np,
                         "sigma": float(info["noise_multiplier"]),
                         "epsilon_spent": float(info["epsilon_spent"])})
    print("split", sp, "done", flush=True)

df = pd.DataFrame(rows)
df.to_csv("results/resample_checks.csv", index=False)


def per_split(arm, col="change"):
    return df[df.arm == arm].groupby("split")[col].mean()


def interval(d, conf):
    ci = stats.t.interval(conf, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return tuple(round(float(v), 3) for v in ci)


say("\nepsilon_spent range:", round(df.epsilon_spent.min(), 3),
    round(df.epsilon_spent.max(), 3))
say("noise multiplier range (same N in every arm):",
    round(df.sigma.min(), 3), round(df.sigma.max(), 3))

say("\nGap change (DP - non-private) by arm; descriptive 95% intervals")
means = {}
for name, *_ in ARMS:
    d = per_split(name)
    means[name] = float(d.mean())
    say(" ", name, "mean", round(d.mean(), 3), "CI", interval(d, 0.95),
        "| non-private gap", round(float(per_split(name, "gap_np").mean()), 3),
        "| per split", [round(float(v), 3) for v in d])

ref = per_split("Ref")
floor_ok = ref.mean() >= 0.02
tag = "" if floor_ok else " [NOT INTERPRETED: P14 floor not met]"
say("\nP14 (floor): Ref mean", round(float(ref.mean()), 3),
    "-> contrasts interpreted" if floor_ok else "-> below 0.02")

c1 = per_split("A1") - per_split("A2")
lo, hi = interval(c1, 0.975)
v15 = "MET" if lo > 0 else ("CONTRADICTED" if hi < 0 else "INCONCLUSIVE")
say("\nC1 = A1 - A2: mean", round(float(c1.mean()), 3), "97.5% CI", (lo, hi),
    "per split", [round(float(v), 3) for v in c1])
say("  P15:", v15 + tag)

c2 = per_split("B2") - per_split("Ref")
lo, hi = interval(c2, 0.975)
v16 = "MET" if hi < 0 else ("CONTRADICTED" if lo > 0 else "INCONCLUSIVE")
say("\nC2 = B2 - Ref: mean", round(float(c2.mean()), 3), "97.5% CI", (lo, hi),
    "per split", [round(float(v), 3) for v in c2])
say("  P16:", v16 + tag)

say("\nP17 (descriptive): A1>=Ref>=A2:",
    means["A1"] >= means["Ref"] >= means["A2"],
    "| Ref>=B1>=B2:", means["Ref"] >= means["B1"] >= means["B2"])

with open("results/resample_output.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")