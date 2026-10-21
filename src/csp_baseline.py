"""CSP + LDA baseline. Fit_predict interface shared by all models:

    fit_predict(X_train, y_train, groups_train, X_test) -> (y_pred, proba_class1, info_dict)

The number of CSP components is chosen by inner GroupKFold over TRAINING subjects only;
the held-out subject is never seen during fitting, scaling or model selection.
"""
import numpy as np
from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import make_pipeline

N_COMPONENTS_GRID = [2, 4, 6, 8]


def make_csp_lda(n_components):
    return make_pipeline(CSP(n_components=n_components, reg="ledoit_wolf", log=True), LDA())


def csp_lda_fit_predict(X_train, y_train, groups_train, X_test, n_inner=5, grid=None):
    X_train = X_train.astype(np.float64)
    X_test = X_test.astype(np.float64)
    n_ch = X_train.shape[1]
    grid = [n for n in (grid or N_COMPONENTS_GRID) if n <= n_ch]

    cv = GroupKFold(n_splits=min(n_inner, len(np.unique(groups_train))))
    best_n, best_score = grid[0], -np.inf
    for n in grid:
        scores = cross_val_score(
            make_csp_lda(n), X_train, y_train, groups=groups_train,
            cv=cv, scoring="balanced_accuracy",
        )
        if scores.mean() > best_score:
            best_n, best_score = n, scores.mean()

    model = make_csp_lda(best_n).fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return pred, proba, {"n_components": int(best_n), "inner_balanced_acc": float(best_score)}
