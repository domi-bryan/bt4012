"""Role A: PyTorch MLP on `LinearPipeline` output (Step 10).

Two ReLU hidden layers (hidden, hidden // 2) with dropout, Adam, mini-batches of 2,048 and
BCEWithLogitsLoss with `pos_weight` "balanced" (negatives / positives of the training rows) or 1 (unweighted). With validation rows,
training stops after `patience` epochs without a better validation PR-AUC and the best epoch's
weights are kept; without them it runs exactly `max_epochs` epochs (the refit in src.experiment).
Runs on CPU so seeded runs are reproducible.
"""
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler
from torch import nn

NAME = "mlp"
EARLY_STOPPING = True   # src.experiment holds out rows for it in the G0-G3 runs
DEFAULTS = {"hidden": 256, "dropout": 0.2, "lr": 1e-3, "batch_size": 2048,
            "max_epochs": 40, "patience": 5, "layers": 2, "weight_decay": 0.0, "pos_weight": "balanced"}
# Random search: 16 of 64 combinations, adding depth (2 or 3 hidden layers) and weight decay
# (widened 2026-11-14; F0-F2 runs used the 8-setting grid over hidden x dropout x lr) and class
# weighting (added 2026-11-16, the same choice the boosting models tune as scale_pos_weight).
GRID = list(ParameterSampler({
    "hidden": [128, 256], "dropout": [0.1, 0.3], "lr": [3e-4, 1e-3],
    "layers": [2, 3], "weight_decay": [0.0, 1e-5], "pos_weight": [1.0, "balanced"],
}, n_iter=16, random_state=0))
PREDICT_BATCH = 65536


def _tensor(X):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(X, dtype="float32")))


def build_net(n_in, hidden, dropout, layers=2):
    """`layers` ReLU hidden layers, each half the width of the one before (hidden, hidden/2, ...)."""
    blocks, width_in = [], n_in
    for i in range(layers):
        width = max(hidden // 2 ** i, 1)
        blocks += [nn.Linear(width_in, width), nn.ReLU(), nn.Dropout(dropout)]
        width_in = width
    return nn.Sequential(*blocks, nn.Linear(width_in, 1))


def _scores(net, X):
    net.eval()
    with torch.no_grad():
        out = [torch.sigmoid(net(X[i:i + PREDICT_BATCH])).squeeze(1)
               for i in range(0, len(X), PREDICT_BATCH)]
    return torch.cat(out).numpy().astype("float64")


class Model:
    def __init__(self, net, history, best_epoch, pos_weight):
        self.net = net
        self.history = history
        # logged in params._fit: per-epoch curves for reports/figures
        self.run_metadata = {"best_epoch": best_epoch, "epochs_run": len(history["loss"]),
                             "pos_weight": round(pos_weight, 6), **history}

    def predict_proba(self, X):
        return _scores(self.net, _tensor(X))


def fit(X, y, X_val=None, y_val=None, params=None, seed=0):
    """Train with early stopping on (X_val, y_val) PR-AUC, or for exactly `max_epochs` epochs without them."""
    if (X_val is None) != (y_val is None):
        raise ValueError("Supply both X_val and y_val for early stopping")
    stop_early = X_val is not None
    p = {**DEFAULTS, **(params or {})}
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    X = _tensor(X)
    X_val = _tensor(X_val) if stop_early else None
    y = torch.from_numpy(np.asarray(y, dtype="float32"))
    y_val = np.asarray(y_val) if stop_early else None
    pos = float(y.sum())
    if pos == 0 or pos == len(y):
        raise ValueError("MLP training requires both classes")
    pos_weight = (len(y) - pos) / pos if p["pos_weight"] == "balanced" else float(p["pos_weight"])

    net = build_net(X.shape[1], p["hidden"], p["dropout"], p["layers"])
    opt = torch.optim.Adam(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))
    history = {"loss": [], "val_pr_auc": []}
    best, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, p["max_epochs"] + 1):
        net.train()
        perm = torch.randperm(len(X), generator=gen)
        total = 0.0
        for i in range(0, len(X), p["batch_size"]):
            idx = perm[i:i + p["batch_size"]]
            opt.zero_grad()
            loss = loss_fn(net(X[idx]).squeeze(1), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        history["loss"].append(round(total / len(X), 6))
        if not stop_early:
            best_epoch = epoch
            continue
        pr_auc = average_precision_score(y_val, _scores(net, X_val))
        history["val_pr_auc"].append(round(float(pr_auc), 6))
        if pr_auc > best:
            best, best_epoch, stale = pr_auc, epoch, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= p["patience"]:
                break
    if stop_early:
        net.load_state_dict(best_state)
    return Model(net, history, best_epoch, pos_weight)


def refit_params(model):
    """The best epoch found by early stopping, as settings for a refit without it."""
    return {"max_epochs": model.run_metadata["best_epoch"]}


def predict_proba(model, X):
    return model.predict_proba(X)


IMPORTANCE = None   # no built-in importance; use permutation importance (src/importance.py)


def feature_importance(model, columns):
    return None
