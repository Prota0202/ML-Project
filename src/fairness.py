import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config


def fairness_table(y_true_score, y_pred_score, sensitive, threshold=config.RISK_THRESHOLD):
    sens = sensitive.fillna("missing").astype(str)
    y_true_bin = (y_true_score < threshold).astype(int).to_numpy()
    y_pred_bin = (y_pred_score < threshold).astype(int)

    rows = []
    for g, idx in sens.groupby(sens).groups.items():
        i = sens.index.get_indexer(idx)
        yt, yp = y_true_bin[i], y_pred_bin[i]
        ytc, ypc = y_true_score.to_numpy()[i], y_pred_score[i]

        tp = int(((yt == 1) & (yp == 1)).sum())
        fn = int(((yt == 1) & (yp == 0)).sum())
        fp = int(((yt == 0) & (yp == 1)).sum())
        tn = int(((yt == 0) & (yp == 0)).sum())

        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)

        rows.append({
            "group": g,
            "n": len(yt),
            "selection_rate(at_risk)": (yp == 1).mean(),
            "accuracy": (yt == yp).mean(),
            "TPR(recall_at_risk)": tpr,
            "FPR": fpr,
            "MAE_score": mean_absolute_error(ytc, ypc),
            "RMSE_score": mean_squared_error(ytc, ypc) ** 0.5,
        })
    return pd.DataFrame(rows).round(3)


def disparity(table, col):
    return float(table[col].max() - table[col].min())
