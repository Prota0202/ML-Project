import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config

# Petites perturbations plausibles, on n'attaque que les champs déclaratifs
HE_DELTAS = [-0.5, -1.0, -1.5, -2.0, -3.0]
HS_DELTAS = [-0.5, -1.0, -2.0]
QUAL_NEW = ["poor"]
METHODE_NEW = ["self-study"]
INTERNET_NEW = ["no"]


@dataclass
class AttackResult:
    base_pred: float
    attack_pred: float
    delta_pred: float
    success: bool
    n_features_changed: int
    changes: dict


def _apply(row, change):
    new = row.copy()
    for k, v in change.items():
        new[k] = v
    return new


def _candidates(row):
    """Treillis : modifications à 1 puis à 2 champs simultanés."""
    out = [{}]
    for d in HE_DELTAS:
        out.append({"heures_etude": max(0.0, float(row.get("heures_etude", 4.0)) + d)})
    for d in HS_DELTAS:
        out.append({"heures_sommeil": max(0.0, float(row.get("heures_sommeil", 7.0)) + d)})
    for v in QUAL_NEW:
        if row.get("qualité_sommeil") != v:
            out.append({"qualité_sommeil": v})
    for v in METHODE_NEW:
        if row.get("méthode_etude") != v:
            out.append({"méthode_etude": v})
    for v in INTERNET_NEW:
        if row.get("accès_internet") != v:
            out.append({"accès_internet": v})

    # Combinaisons à 2
    base = list(out)
    pairs = []
    for c1, c2 in itertools.combinations(base[1:], 2):
        if not (set(c1) & set(c2)):  # deux clés différentes seulement
            merged = dict(c1)
            merged.update(c2)
            pairs.append(merged)
    return base + pairs


def evade(pipe, row, threshold=config.RISK_THRESHOLD):
    """Cherche la perturbation minimale (en nb de champs) qui passe sous le seuil."""
    base_pred = float(pipe.predict(pd.DataFrame([row]))[0])

    best = None
    for change in _candidates(row):
        if not change:
            continue
        new_row = _apply(row, change)
        pred = float(pipe.predict(pd.DataFrame([new_row]))[0])
        if pred >= threshold:
            continue
        n_changed = len(change)
        cand = AttackResult(
            base_pred=base_pred,
            attack_pred=pred,
            delta_pred=pred - base_pred,
            success=True,
            n_features_changed=n_changed,
            changes={k: (row[k], v) for k, v in change.items()},
        )
        if best is None or cand.n_features_changed < best.n_features_changed:
            best = cand

    if best is not None:
        return best
    return AttackResult(base_pred, base_pred, 0.0, False, 0, {})


def evaluate_attack_success(pipe, X_safe, n=200):
    base_preds = pipe.predict(X_safe)
    safe_idx = np.where(base_preds >= config.RISK_THRESHOLD)[0][:n]

    n_success = 0
    changes_when_ok = []
    for i in safe_idx:
        res = evade(pipe, X_safe.iloc[i])
        if res.success:
            n_success += 1
            changes_when_ok.append(res.n_features_changed)

    return {
        "n_evaluated": int(len(safe_idx)),
        "n_success": int(n_success),
        "success_rate": n_success / max(1, len(safe_idx)),
        "mean_changes_on_success": (
            float(np.mean(changes_when_ok)) if changes_when_ok else None
        ),
    }
