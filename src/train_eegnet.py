import copy
import time

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

# ---------------------------------------------------------------------
# LOSO interface (used by src/subject_independent.run_loso)
# Settings fixed before any EEGNet LOSO result existed (see README, Methods).
# ---------------------------------------------------------------------

LOSO_SEED = 42
LOSO_BATCH_SIZE = 64
LOSO_MAX_EPOCHS = 100
LOSO_PATIENCE = 10
LOSO_LEARNING_RATE = 1e-3
LOSO_N_VAL_SUBJECTS = 4


def get_device():
    # CPU only (fixed with the other settings): on MPS (torch 2.10.0) EEGNet failed to fit even 256
    # training trials, while the same code and seed overfit them on CPU.
    return torch.device("cpu")


def _predict_logits(model, X, device, batch_size=256):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            out.append(model(X[i:i + batch_size].to(device)).cpu())
    return torch.cat(out)


def eegnet_fit_predict(X_train, y_train, groups_train, X_test, input_scale):
    """fit_predict(X_train, y_train, groups_train, X_test) -> (pred, proba_class1, info).

    Bind input_scale with functools.partial: a FIXED constant, not a fitted statistic, applied
    identically to train and test. 1e6 for raw data (volts -> microvolts); 1.0 for Euclidean-
    aligned data, which is already dimensionless and ~unit scale.

    Validation (for early stopping only) = LOSO_N_VAL_SUBJECTS TRAINING subjects held out by
    group. The test subject is used only for the final forward pass.
    """
    device = get_device()
    start = time.time()

    # Validation subjects: seeded choice among training subjects only
    rng = np.random.default_rng(LOSO_SEED)
    train_subjects = np.unique(groups_train)
    val_subjects = np.sort(rng.choice(train_subjects, LOSO_N_VAL_SUBJECTS, replace=False))
    val_mask = np.isin(groups_train, val_subjects)

    def to_tensor(a):
        return torch.tensor(a[:, None, :, :] * input_scale, dtype=torch.float32)

    X_fit, y_fit = to_tensor(X_train[~val_mask]), torch.tensor(y_train[~val_mask], dtype=torch.long)
    X_val, y_val = to_tensor(X_train[val_mask]), torch.tensor(y_train[val_mask], dtype=torch.long)
    X_te = to_tensor(X_test)

    generator = torch.Generator().manual_seed(LOSO_SEED)
    loader = DataLoader(TensorDataset(X_fit, y_fit), batch_size=LOSO_BATCH_SIZE,
                        shuffle=True, generator=generator)

    torch.manual_seed(LOSO_SEED)
    model = EEGNet(X_fit.shape[2], X_fit.shape[3]).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LOSO_LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val, best_epoch, best_state = np.inf, 0, None

    for epoch in range(LOSO_MAX_EPOCHS):
        model.train()
        total, n = 0.0, 0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            model.apply_max_norm()
            total += loss.item() * len(batch_y)
            n += len(batch_y)
        train_losses.append(total / n)

        val_loss = loss_function(_predict_logits(model, X_val, device), y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
        elif epoch + 1 - best_epoch >= LOSO_PATIENCE:
            break

    model.load_state_dict(best_state)
    proba = torch.softmax(_predict_logits(model, X_te, device), dim=1)[:, 1].numpy()
    pred = (proba >= 0.5).astype(int)

    info = {
        "device": str(device),
        "val_subjects": "-".join(str(int(s)) for s in val_subjects),
        "epochs_run": len(train_losses),
        "best_epoch": int(best_epoch),
        "best_val_loss": float(best_val),
        "input_scale": float(input_scale),
        "train_time_s": round(time.time() - start, 1),
        "train_losses": [float(v) for v in train_losses],
        "val_losses": [float(v) for v in val_losses],
    }
    return pred, proba, info
