from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


def load_train(path=None, nrows=None):
    path = path or config.TRAIN_CSV
    if not Path(path).is_file():
        raise FileNotFoundError(
            f"Fichier introuvable : {path}\n"
            "→ place le dossier student_dataset sous data/ ou définis SMARTSCHOOL_DATA"
        )
    df = pd.read_csv(path, nrows=nrows)
    df = df.drop(columns=[config.ID_COL], errors="ignore")
    return df


def add_at_risk_label(df, threshold=config.RISK_THRESHOLD):
    out = df.copy()
    out["at_risk"] = (out[config.TARGET] < threshold).astype(int)
    return out


def make_splits(df, test_size=0.15, val_size=0.15, random_state=config.RANDOM_STATE,
                features=None, target=config.TARGET, stratify_by_at_risk=True):
    """train ~70% / val ~15% / test ~15%, stratifié sur at_risk."""
    feats = features or config.FEATURES_FINAL
    X = df[feats].copy()
    y = df[target].copy()
    strat = (y < config.RISK_THRESHOLD).astype(int) if stratify_by_at_risk else None

    X_tv, X_test, y_tv, y_test, strat_tv, _ = train_test_split(
        X, y, strat, test_size=test_size, random_state=random_state, stratify=strat,
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=rel_val, random_state=random_state, stratify=strat_tv,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
