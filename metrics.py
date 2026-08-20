import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score, accuracy_score

def binary_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1-1e-9)
    out = {
        "brier": float(brier_score_loss(y,p)),
        "log_loss": float(log_loss(y, np.c_[1-p,p], labels=[0,1])),
        "accuracy_0.5": float(accuracy_score(y,(p>=0.5).astype(int))),
    }
    if len(np.unique(y)) > 1:
        out["roc_auc"] = float(roc_auc_score(y,p))
        out["average_precision"] = float(average_precision_score(y,p))
    else:
        out["roc_auc"] = None
        out["average_precision"] = None
    return out

def stratify(y, p, values):
    values = np.asarray(values, dtype=float)
    q = np.quantile(values, 0.5)
    low = values <= q
    high = values > q
    out = {}
    if low.any():
        out["low"] = binary_metrics(np.asarray(y)[low], np.asarray(p)[low])
    if high.any():
        out["high"] = binary_metrics(np.asarray(y)[high], np.asarray(p)[high])
    return out
