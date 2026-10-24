"""Leave-one-subject-out evaluation shared by every model (KNN, CSP-LDA, EEGNet).

Every model sees identical folds: for each held-out subject, train on all other subjects.
Per-subject predictions are saved as they finish, so a crashed run can resume.
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)

from src.alignment import align_by_subject
from src.utils import set_seed

LABELS = [0, 1]


def _clean(n):
    return n.strip(".").upper()


def select_channels(X, ch_names, channels):
    clean = [_clean(c) for c in ch_names]
    idx = [clean.index(_clean(c)) for c in channels]
    return X[:, idx, :], [ch_names[i] for i in idx]


def compute_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=LABELS).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_loso(name, fit_predict, X, y, groups, ch_names, align=False, channels=None,
             out_dir="results", seed=42, resume=True, subjects=None):
    """Run LOSO for one model/condition. Returns a per-subject metrics DataFrame.

    subjects: optional subset of held-out subjects (for quick tests). Each fold still trains
    on ALL other subjects, so folds are identical to a full run.
    """
    if channels is not None:
        X, ch_names = select_channels(X, ch_names, channels)
    if align:
        X = align_by_subject(X, groups)      # per-subject, label-free

    tag = f"{name}_{'ea' if align else 'raw'}_{'full' if channels is None else '-'.join(channels)}"
    pred_dir = os.path.join(out_dir, "predictions", tag)
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "tables"), exist_ok=True)

    rows = []
    for s in (np.unique(groups) if subjects is None else subjects):
        path = os.path.join(pred_dir, f"S{s}.npz")
        if resume and os.path.exists(path):
            d = np.load(path, allow_pickle=True)
            y_true, y_pred, info = d["y_true"], d["y_pred"], json.loads(str(d["info"]))
        else:
            set_seed(seed)
            test = groups == s
            train = ~test
            y_pred, proba, info = fit_predict(X[train], y[train], groups[train], X[test])
            y_true = y[test]
            np.savez(path, y_true=y_true, y_pred=y_pred, proba=proba, info=json.dumps(info))

        # Lists (e.g. EEGNet loss curves) stay in the .npz, not in the per-subject table
        scalars = {k: v for k, v in info.items() if not isinstance(v, (list, dict))}
        row = {"subject": int(s), "n_trials": len(y_true), **compute_metrics(y_true, y_pred), **scalars}
        rows.append(row)
        print(f"[{tag}] S{s}: acc={row['accuracy']:.3f} bal_acc={row['balanced_accuracy']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "tables", f"{tag}_per_subject.csv"), index=False)
    print(f"[{tag}] mean +/- std  acc={df.accuracy.mean():.3f}+/-{df.accuracy.std():.3f}  "
          f"macroF1={df.macro_f1.mean():.3f}+/-{df.macro_f1.std():.3f}  "
          f"bal_acc={df.balanced_accuracy.mean():.3f}+/-{df.balanced_accuracy.std():.3f}")
    return df