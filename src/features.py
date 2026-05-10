import pandas as pd

_SLEEP_MAP = {"good": 1.0, "average": 0.6, "poor": 0.3}


def add_derived(df):
    out = df.copy()

    methode = out["méthode_etude"].fillna("missing").astype(str)
    encadree = methode.isin(["coaching", "mixed"]).astype(int)
    he = out["heures_etude"].fillna(out["heures_etude"].median())
    out["study_efficiency"] = he * encadree

    qmap = out["qualité_sommeil"].fillna("average").astype(str).map(_SLEEP_MAP).fillna(0.6)
    out["sleep_score"] = out["heures_sommeil"] * qmap

    # NOTE: cette feature n'apporte rien (importance ~3e-5 en val).
    # Je la laisse pour discussion dans le rapport, mais à virer dans une v2.
    internet_no = (out["accès_internet"].fillna("missing").astype(str) == "no").astype(int)
    eval_low = (out["évaluation_établissement"].astype(str) == "low").astype(int)
    out["risk_internet_low"] = internet_no * eval_low

    return out


DERIVED_NUM = ["study_efficiency", "sleep_score", "risk_internet_low"]
