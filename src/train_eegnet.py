import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from .eegnet import EEGNet


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


def run_eegnet(X, y, groups):

    # -----------------------------------------
    # Hold out one subject for final testing
    # -----------------------------------------

    unique_subjects = np.unique(groups)

    test_subject = unique_subjects[-1]

    train_mask = groups != test_subject
    test_mask = groups == test_subject

    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    groups_train = groups[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    # -----------------------------------------
    # Hold out another subject for validation
    # -----------------------------------------

    validation_subject = np.unique(groups_train)[-1]

    train_mask_2 = groups_train != validation_subject
    validation_mask = groups_train == validation_subject

    X_train = X_train_full[train_mask_2]
    y_train = y_train_full[train_mask_2]

    X_val = X_train_full[validation_mask]
    y_val = y_train_full[validation_mask]

    print("\nEEGNet data split:")
    print("Training subject count:",
          len(np.unique(groups_train[train_mask_2])))
    print("Validation subject:",
          validation_subject)
    print("Test subject:",
          test_subject)

    # -----------------------------------------
    # Add channel dimension
    # -----------------------------------------

    X_train = X_train[:, None, :, :]
    X_val = X_val[:, None, :, :]
    X_test = X_test[:, None, :, :]

    # -----------------------------------------
    # Convert to tensors
    # -----------------------------------------

    X_train = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    X_val = torch.tensor(
        X_val,
        dtype=torch.float32
    )

    X_test = torch.tensor(
        X_test,
        dtype=torch.float32
    )

    y_train = torch.tensor(
        y_train,
        dtype=torch.long
    )

    y_val = torch.tensor(
        y_val,
        dtype=torch.long
    )

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    # -----------------------------------------
    # DataLoader
    # -----------------------------------------

    dataset = TensorDataset(
        X_train,
        y_train
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # -----------------------------------------
    # Create EEGNet
    # -----------------------------------------

    channels = X_train.shape[2]
    time_points = X_train.shape[3]

    model = EEGNet(
        channels,
        time_points
    )

    # -----------------------------------------
    # Loss and optimizer
    # -----------------------------------------

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    train_losses = []
    validation_losses = []

    # -----------------------------------------
    # Training
    # -----------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            output = model(batch_X)

            loss = loss_function(
                output,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_train_loss = (
            total_loss / len(loader)
        )

        train_losses.append(
            average_train_loss
        )

        # -------------------------------------
        # Validation
        # -------------------------------------

        model.eval()

        with torch.no_grad():

            validation_output = model(
                X_val
            )

            validation_loss = loss_function(
                validation_output,
                y_val
            )

        validation_losses.append(
            validation_loss.item()
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Train Loss: "
            f"{average_train_loss:.4f} "
            f"Validation Loss: "
            f"{validation_loss.item():.4f}"
        )

    # -----------------------------------------
    # Final test on completely unseen subject
    # -----------------------------------------

    model.eval()

    with torch.no_grad():

        output = model(X_test)

        predictions = torch.argmax(
            output,
            dim=1
        )

    predictions = predictions.numpy()
    actual = y_test.numpy()

    # -----------------------------------------
    # Evaluation
    # -----------------------------------------

    print("\nEEGNet Results")

    print(
        "Accuracy:",
        accuracy_score(
            actual,
            predictions
        )
    )

    print(
        "Precision:",
        precision_score(
            actual,
            predictions,
            zero_division=0
        )
    )

    print(
        "Recall:",
        recall_score(
            actual,
            predictions,
            zero_division=0
        )
    )

    print(
        "F1:",
        f1_score(
            actual,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            actual,
            predictions
        )
    )

    return {
        "model": model,
        "train_losses": train_losses,
        "validation_losses": validation_losses,
        "actual": actual,
        "predictions": predictions
    }