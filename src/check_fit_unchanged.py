"""Check that fit() still reproduces a stored part 1 row after the optimizer option was added."""
import warnings
import pandas as pd
from common import load_adult, make_split, fit, scores, group_metrics

warnings.filterwarnings("ignore", category=UserWarning)
df = pd.read_csv("results/multisplit.csv")
row = df[(df.split == 1) & (df.seed == 0) & (df.eps == 0.1)].iloc[0]
X, y = load_adult()
split = make_split(X, y, 1)
model, info = fit(split, 0, eps=0.1)
m = group_metrics(scores(model, split["Xte"]), split)
keys = ["accuracy", "tpr_M", "tpr_F", "tpr_gap"]
for k in keys:
    print(k, "stored", round(float(row[k]), 6), "now", round(float(m[k]), 6))
print("reproduces:", all(abs(float(row[k]) - float(m[k])) < 1e-9 for k in keys))
