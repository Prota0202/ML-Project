from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .preprocessing import build_preprocessor


def make_sklearn_models(seed=config.RANDOM_STATE):
    pre = build_preprocessor

    return {
        "baseline_mean": Pipeline([("pre", pre()), ("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([("pre", pre()), ("model", Ridge(alpha=1.0))]),
        "random_forest": Pipeline([
            ("pre", pre(scale_numeric=False)),
            ("model", RandomForestRegressor(
                n_estimators=200, max_depth=None, min_samples_leaf=20,
                n_jobs=-1, random_state=seed,
            )),
        ]),
        "hist_gbrt": Pipeline([
            ("pre", pre(scale_numeric=False)),
            ("model", HistGradientBoostingRegressor(
                max_depth=8, learning_rate=0.06, max_iter=400,
                min_samples_leaf=80, l2_regularization=1e-3, random_state=seed,
            )),
        ]),
    }


@dataclass
class RegMetrics:
    rmse_train: float
    rmse_val: float
    rmse_test: float
    mae_test: float
    r2_test: float
    acc_at_risk_test: float
    f1_at_risk_test: float


def regression_metrics(pipe, X_train, y_train, X_val, y_val, X_test, y_test,
                       threshold=config.RISK_THRESHOLD):
    p_tr = pipe.predict(X_train)
    p_va = pipe.predict(X_val)
    p_te = pipe.predict(X_test)
    return RegMetrics(
        rmse_train=mean_squared_error(y_train, p_tr) ** 0.5,
        rmse_val=mean_squared_error(y_val, p_va) ** 0.5,
        rmse_test=mean_squared_error(y_test, p_te) ** 0.5,
        mae_test=mean_absolute_error(y_test, p_te),
        r2_test=r2_score(y_test, p_te),
        acc_at_risk_test=accuracy_score(y_test < threshold, p_te < threshold),
        f1_at_risk_test=f1_score(y_test < threshold, p_te < threshold),
    )


def metrics_to_df(name_to_metrics):
    rows = []
    for n, m in name_to_metrics.items():
        d = asdict(m)
        d["model"] = n
        rows.append(d)
    cols = ["model", "rmse_train", "rmse_val", "rmse_test", "mae_test",
            "r2_test", "acc_at_risk_test", "f1_at_risk_test"]
    return pd.DataFrame(rows)[cols].round(4)


# MLP PyTorch - on a essayé plus simple (128-64-32, BN, ReLU, MSE) mais ça
# plafonnait à RMSE 11. La version actuelle gagne 2 points.

class TabularMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256, 128, 64), p_drop=0.10):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(p_drop)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@dataclass
class MLPLog:
    train_rmse: list
    val_rmse: list


def train_mlp(X_train, y_train, X_val, y_val, epochs=60, batch_size=2048,
              lr=3e-3, weight_decay=1e-3, hidden=(256, 128, 64),
              seed=config.RANDOM_STATE, device=None, verbose=True, patience=8):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    Xt = torch.tensor(X_train, dtype=torch.float32, device=device)
    yt = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xv = torch.tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.tensor(y_val, dtype=torch.float32, device=device)

    loader = DataLoader(TensorDataset(Xt, yt), batch_size=batch_size, shuffle=True)
    model = TabularMLP(in_dim=X_train.shape[1], hidden=hidden).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss(beta=2.0)

    log = MLPLog([], [])
    best_state, best_val, no_improve = None, float("inf"), 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            tr_rmse = float(torch.mean((model(Xt) - yt) ** 2) ** 0.5)
            va_rmse = float(torch.mean((model(Xv) - yv) ** 2) ** 0.5)
        log.train_rmse.append(tr_rmse)
        log.val_rmse.append(va_rmse)

        if va_rmse < best_val - 1e-3:
            best_val = va_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if verbose and (epoch % 5 == 0 or epoch == epochs - 1):
            lr_now = sched.get_last_lr()[0]
            print(f"  epoch {epoch:02d}  train RMSE={tr_rmse:.3f}  val RMSE={va_rmse:.3f}  lr={lr_now:.2e}")

        if no_improve >= patience:
            if verbose:
                print(f"  early stop @ epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, log


@torch.no_grad()
def mlp_predict(model, X, device=None):
    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.eval()
    return model(torch.tensor(X, dtype=torch.float32, device=device)).cpu().numpy()
