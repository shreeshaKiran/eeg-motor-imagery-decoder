import numpy as np

from mne.decoding import CSP
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)


def run_csp_subject_independent(X, y, groups):
    """
    X: trials x channels x time
    y: integer class labels, 0 and 1
    groups: subject ID for every trial
    """

    results = []
    all_actual = []
    all_predictions = []

    for test_subject in np.unique(groups):
        train_mask = groups != test_subject
        test_mask = groups == test_subject

        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]

        model = Pipeline([
            (
                "csp",
                CSP(
                    n_components=4,
                    reg=None,
                    log=True,
                    norm_trace=False,
                ),
            ),
            (
                "lda",
                LinearDiscriminantAnalysis(),
            ),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        result = {
            "subject": int(test_subject),
            "accuracy": accuracy_score(
                y_test, predictions
            ),
            "macro_f1": f1_score(
                y_test,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                y_test, predictions
            ),
        }

        results.append(result)
        all_actual.extend(y_test)
        all_predictions.extend(predictions)

        print(result)

    actual = np.asarray(all_actual)
    predictions = np.asarray(all_predictions)

    print("\nMean subject accuracy:",
          np.mean([r["accuracy"] for r in results]))
    print("Mean subject macro-F1:",
          np.mean([r["macro_f1"] for r in results]))
    print("Mean subject balanced accuracy:",
          np.mean([r["balanced_accuracy"] for r in results]))

    return {
        "subject_results": results,
        "actual": actual,
        "predictions": predictions,
    }