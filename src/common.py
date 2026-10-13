"""Shared code for the DP-fairness experiments on UCI Adult.

Functions are taken from the exploratory notebook; behaviour is meant to be identical.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from opacus import PrivacyEngine

DELTA = 1e-5


def load_adult():
    data = fetch_openml("adult", version=2, as_frame=True)
    X, y = data.data, (data.target == ">50K").astype(int)
    return X, y


def build_preprocessor(X):
    num = X.select_dtypes(include="number").columns
    cat = X.select_dtypes(exclude="number").columns
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])


def _dense(a):
    return a.toarray() if hasattr(a, "toarray") else a


def make_split(X, y, split_seed):
    """70/30 stratified split; preprocessing is fitted on the training part only."""
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=split_seed, stratify=y)
    pre = build_preprocessor(X).fit(Xtr)
    to_t = lambda a: torch.tensor(_dense(a), dtype=torch.float32)
    return {
        "Xtr": to_t(pre.transform(Xtr)),
        "ytr": torch.tensor(ytr.values, dtype=torch.float32),
        "Xte": to_t(pre.transform(Xte)),
        "yte": yte.values,
        "male": (Xte["sex"] == "Male").values,
    }


def fit(split, seed, eps=None, clip=1.0, noise_multiplier=None,
        epochs=10, bs=512, lr=0.01):
    """Non-private if eps and noise_multiplier are None; DP-SGD to a target epsilon
    if eps is given; clipping with a fixed noise multiplier if noise_multiplier is given."""
    Xt, yt = split["Xtr"], split["ytr"]
    torch.manual_seed(seed)
    model = nn.Linear(Xt.shape[1], 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Xt, yt), batch_size=bs, shuffle=True)
    pe = None
    if noise_multiplier is not None:
        pe = PrivacyEngine()
        model, opt, loader = pe.make_private(
            module=model, optimizer=opt, data_loader=loader,
            noise_multiplier=noise_multiplier, max_grad_norm=clip)
    elif eps is not None:
        pe = PrivacyEngine()
        model, opt, loader = pe.make_private_with_epsilon(
            module=model, optimizer=opt, data_loader=loader,
            target_epsilon=eps, target_delta=DELTA, epochs=epochs, max_grad_norm=clip)
    loss_fn = nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb).squeeze(1), yb).backward()
            opt.step()
    model.eval()
    info = {}
    if eps is not None and noise_multiplier is None:
        info = {"noise_multiplier": opt.noise_multiplier,
                "epsilon_spent": pe.get_epsilon(DELTA)}
    return model, info


def scores(model, Xe):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(Xe).squeeze(1)).numpy()


def group_metrics(s, split, threshold=0.5):
    ye, male = split["yte"], split["male"]
    p = (s > threshold).astype(int)
    out = {"accuracy": (p == ye).mean(),
           "auc_M": roc_auc_score(ye[male], s[male]),
           "auc_F": roc_auc_score(ye[~male], s[~male])}
    for name, m in [("M", male), ("F", ~male)]:
        yg, pg = ye[m], p[m]
        out[f"sel_{name}"] = pg.mean()
        out[f"tpr_{name}"] = pg[yg == 1].mean()
        out[f"fpr_{name}"] = pg[yg == 0].mean()
    out["sel_gap"] = out["sel_M"] - out["sel_F"]
    out["tpr_gap"] = out["tpr_M"] - out["tpr_F"]
    out["fpr_gap"] = out["fpr_M"] - out["fpr_F"]
    return out


def matched_tpr_gap(s, split, rate):
    """TPR gap after thresholding so that the overall selection rate equals `rate`."""
    ye, male = split["yte"], split["male"]
    p = (s > np.quantile(s, 1 - rate)).astype(int)
    pos_m, pos_f = male & (ye == 1), ~male & (ye == 1)
    return p[pos_m].mean() - p[pos_f].mean()


if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    X, y = load_adult()
    split = make_split(X, y, 0)
    show = lambda name, m: print(name, {k: round(float(v), 3) for k, v in m.items()})

    model, _ = fit(split, seed=0)
    show("baseline seed 0:", group_metrics(scores(model, split["Xte"]), split))

    model, info = fit(split, seed=0, eps=1.0)
    print("dp eps=1 info:", {k: round(float(v), 3) for k, v in info.items()})
    show("dp eps=1 seed 0:", group_metrics(scores(model, split["Xte"]), split))