import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
)

from .eegnet import EEGNet


BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001


def run_subject_independent(X, y, groups):

    subjects = np.unique(groups)

    print("\nSubject-independent EEGNet evaluation")

    results = []

    for test_subject in subjects:

        print(f"\nHeld-out subject: {test_subject}")

        train_mask = groups != test_subject
        test_mask = groups == test_subject

        X_train = X[train_mask][:, None, :, :]
        X_test = X[test_mask][:, None, :, :]

        y_train = y[train_mask]
        y_test = y[test_mask]

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)

        loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=BATCH_SIZE,
            shuffle=True,
        )

        model = EEGNet(
            channels=X_train.shape[2],
            time_points=X_train.shape[3],
            num_classes=2,
        )

        loss_function = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        for epoch in range(EPOCHS):

            model.train()

            for batch_X, batch_y in loader:

                optimizer.zero_grad()

                output = model(batch_X)
                loss = loss_function(output, batch_y)

                loss.backward()
                optimizer.step()

        model.eval()

        with torch.no_grad():
            output = model(X_test)
            predictions = output.argmax(dim=1).cpu().numpy()

        actual = y_test.cpu().numpy()

        result = {
            "subject": int(test_subject),
            "accuracy": accuracy_score(actual, predictions),
            "macro_f1": f1_score(
                actual,
                predictions,
                average="macro",
                zero_division=0,
            ),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predictions
            ),
        }

        results.append(result)

        print(
            f"Accuracy={result['accuracy']:.4f}, "
            f"Macro-F1={result['macro_f1']:.4f}, "
            f"Balanced accuracy={result['balanced_accuracy']:.4f}"
        )

    print("\nMean subject-level EEGNet metrics:")

    for metric in ("accuracy", "macro_f1", "balanced_accuracy"):
        values = [r[metric] for r in results]
        print(
            f"{metric}: {np.mean(values):.4f} "
            f"+/- {np.std(values):.4f}"
        )

    return results