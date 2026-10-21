import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

from .features import extract_features


SAMPLING_RATE = 160


def run_knn(X, y, groups):

    print("\nKNN baseline")

    features = extract_features(X, SAMPLING_RATE)

    logo = LeaveOneGroupOut()

    subject_results = []
    all_actual = []
    all_predictions = []

    for train_idx, test_idx in logo.split(
        features, y, groups
    ):

        test_subject = np.unique(groups[test_idx])[0]

        X_train = features[train_idx]
        X_test = features[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=5)),
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        result = {
            "subject": int(test_subject),
            "accuracy": accuracy_score(y_test, predictions),
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

        subject_results.append(result)
        all_actual.extend(y_test)
        all_predictions.extend(predictions)

        print(
            f"Subject {test_subject}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Macro-F1={result['macro_f1']:.4f}, "
            f"Balanced accuracy={result['balanced_accuracy']:.4f}"
        )

    print("\nMean subject-level KNN metrics:")
    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = [r[metric] for r in subject_results]
        print(f"{metric}: {np.mean(values):.4f} "
              f"+/- {np.std(values):.4f}")

    return {
        "subject_results": subject_results,
        "actual": np.asarray(all_actual),
        "predictions": np.asarray(all_predictions),
    }