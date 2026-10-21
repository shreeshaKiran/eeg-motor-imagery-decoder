import numpy as np
import matplotlib.pyplot as plt
import mne

from mne.datasets import eegbci
from mne.io import read_raw_edf, concatenate_raws

from sklearn.model_selection import LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# SETTINGS
# ============================================================

SUBJECTS = list(range(1, 11))

RUNS = [4, 8, 12]

LOW_FREQ = 8
HIGH_FREQ = 30

EPOCH_START = 1.0
EPOCH_END = 4.0

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001


# ============================================================
# LOAD ONE SUBJECT
# ============================================================

def load_subject(subject):

    print(f"Loading subject {subject}...")

    file_paths = eegbci.load_data(
        subjects=subject,
        runs=RUNS
    )

    raw_files = []

    for file_path in file_paths:

        raw = read_raw_edf(
            file_path,
            preload=True,
            verbose=False
        )

        raw_files.append(raw)

    raw = concatenate_raws(raw_files)

    # Make channel names consistent
    eegbci.standardize(raw)

    return raw


# ============================================================
# PREPROCESS EEG
# ============================================================

def preprocess_subject(raw):

    # Keep EEG channels only
    raw.pick("eeg")

    # Bandpass filter
    raw.filter(
        l_freq=LOW_FREQ,
        h_freq=HIGH_FREQ,
        verbose=False
    )

    # Average reference
    raw.set_eeg_reference("average", verbose=False)

    return raw


# ============================================================
# CREATE EPOCHS
# ============================================================

def create_epochs(raw):

    events, event_dict = mne.events_from_annotations(
        raw,
        verbose=False
    )

    print("Events:", event_dict)

    # The PhysioNet motor imagery dataset uses:
    #
    # T1 = left fist
    # T2 = right fist
    #
    # We only want these two classes.

    selected_events = {}

    if "T1" in event_dict:
        selected_events["left"] = event_dict["T1"]

    if "T2" in event_dict:
        selected_events["right"] = event_dict["T2"]

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_events,
        tmin=EPOCH_START,
        tmax=EPOCH_END,
        baseline=None,
        preload=True,
        verbose=False
    )

    return epochs


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    all_X = []
    all_y = []
    all_groups = []

    for subject in SUBJECTS:

        raw = load_subject(subject)

        raw = preprocess_subject(raw)

        epochs = create_epochs(raw)

        X = epochs.get_data()

        # Convert labels
        y = epochs.events[:, -1]

        # Convert event IDs into 0/1
        unique_labels = np.unique(y)

        label_map = {
            unique_labels[0]: 0,
            unique_labels[1]: 1
        }

        y = np.array([
            label_map[label]
            for label in y
        ])

        all_X.append(X)
        all_y.append(y)

        # Keep track of which subject each trial belongs to
        all_groups.extend(
            [subject] * len(y)
        )

        print(
            f"Subject {subject}: "
            f"{len(y)} trials"
        )

    X = np.concatenate(all_X)

    y = np.concatenate(all_y)

    groups = np.array(all_groups)

    print("\nFinal dataset:")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Subjects:", np.unique(groups))

    return X, y, groups


# ============================================================
# CREATE FEATURES FOR KNN
# ============================================================

def extract_features(X):

    features = []

    for trial in X:

        # trial shape:
        # channels x time

        mean = np.mean(trial, axis=1)

        std = np.std(trial, axis=1)

        variance = np.var(trial, axis=1)

        maximum = np.max(trial, axis=1)

        minimum = np.min(trial, axis=1)

        trial_features = np.concatenate([
            mean,
            std,
            variance,
            maximum,
            minimum
        ])

        features.append(trial_features)

    return np.array(features)


# ============================================================
# KNN BASELINE
# ============================================================

def run_knn(X, y, groups):

    print("\n==============================")
    print("KNN BASELINE")
    print("==============================")

    features = extract_features(X)

    logo = LeaveOneGroupOut()

    predictions = []
    actual = []

    for train_index, test_index in logo.split(
        features,
        y,
        groups
    ):

        X_train = features[train_index]
        X_test = features[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        model = Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "knn",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            )
        ])

        model.fit(
            X_train,
            y_train
        )

        prediction = model.predict(X_test)

        predictions.extend(prediction)
        actual.extend(y_test)

    predictions = np.array(predictions)
    actual = np.array(actual)

    print_metrics(
        actual,
        predictions,
        "KNN"
    )

    return actual, predictions


# ============================================================
# EEGNET MODEL
# ============================================================

class EEGNet(nn.Module):

    def __init__(
        self,
        channels,
        time_points,
        num_classes=2
    ):

        super().__init__()

        self.temporal = nn.Conv2d(
            1,
            16,
            kernel_size=(1, 64),
            padding=(0, 32),
            bias=False
        )

        self.batch_norm1 = nn.BatchNorm2d(16)

        self.spatial = nn.Conv2d(
            16,
            32,
            kernel_size=(channels, 1),
            groups=16,
            bias=False
        )

        self.batch_norm2 = nn.BatchNorm2d(32)

        self.activation = nn.ELU()

        self.pool = nn.AvgPool2d(
            kernel_size=(1, 4)
        )

        self.dropout = nn.Dropout(
            0.5
        )

        # Determine flattened size automatically
        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                channels,
                time_points
            )

            dummy = self.forward_features(
                dummy
            )

            flattened_size = dummy.view(
                1,
                -1
            ).shape[1]

        self.classifier = nn.Linear(
            flattened_size,
            num_classes
        )

    def forward_features(self, x):

        x = self.temporal(x)

        x = self.batch_norm1(x)

        x = self.spatial(x)

        x = self.batch_norm2(x)

        x = self.activation(x)

        x = self.pool(x)

        x = self.dropout(x)

        return x

    def forward(self, x):

        x = self.forward_features(x)

        x = x.flatten(
            start_dim=1
        )

        x = self.classifier(x)

        return x


# ============================================================
# TRAIN EEGNET
# ============================================================

def train_eegnet(
    X_train,
    y_train,
    X_test,
    y_test
):

    print("\n==============================")
    print("EEGNET")
    print("==============================")

    # Convert to PyTorch format
    #
    # Current:
    # trials x channels x time
    #
    # Needed:
    # trials x 1 x channels x time

    X_train = X_train[:, np.newaxis, :, :]
    X_test = X_test[:, np.newaxis, :, :]

    X_train = torch.tensor(
        X_train,
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

    y_test = torch.tensor(
        y_test,
        dtype=torch.long
    )

    train_dataset = TensorDataset(
        X_train,
        y_train
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    channels = X_train.shape[2]

    time_points = X_train.shape[3]

    model = EEGNet(
        channels,
        time_points
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    training_losses = []

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0

        for batch_X, batch_y in train_loader:

            optimizer.zero_grad()

            output = model(batch_X)

            loss = loss_function(
                output,
                batch_y
            )

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss /
            len(train_loader)
        )

        training_losses.append(
            average_loss
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss: {average_loss:.4f}"
        )

    # Evaluate
    model.eval()

    with torch.no_grad():

        output = model(X_test)

        predictions = torch.argmax(
            output,
            dim=1
        )

    predictions = predictions.numpy()

    actual = y_test.numpy()

    print_metrics(
        actual,
        predictions,
        "EEGNet"
    )

    return (
        model,
        training_losses,
        actual,
        predictions
    )


# ============================================================
# METRICS
# ============================================================

def print_metrics(
    actual,
    predictions,
    model_name
):

    accuracy = accuracy_score(
        actual,
        predictions
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0
    )

    print(f"\n{model_name} Results")

    print(
        f"Accuracy:  {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall:    {recall:.4f}"
    )

    print(
        f"F1 Score:  {f1:.4f}"
    )

    cm = confusion_matrix(
        actual,
        predictions
    )

    print("\nConfusion Matrix:")
    print(cm)


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "EEG Motor Imagery Neural Decoder"
    )

    # Load EEG
    X, y, groups = load_dataset()

    # ----------------------------------------
    # KNN
    # ----------------------------------------

    knn_actual, knn_predictions = run_knn(
        X,
        y,
        groups
    )

    # ----------------------------------------
    # EEGNet
    # ----------------------------------------

    # For the first version, hold out the
    # final subject as the test subject.

    test_subject = SUBJECTS[-1]

    train_mask = groups != test_subject
    test_mask = groups == test_subject

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(
        f"\nTraining subjects: "
        f"{SUBJECTS[:-1]}"
    )

    print(
        f"Test subject: "
        f"{test_subject}"
    )

    model, losses, actual, predictions = train_eegnet(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ----------------------------------------
    # Plot training loss
    # ----------------------------------------

    plt.figure()

    plt.plot(
        losses
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Loss"
    )

    plt.title(
        "EEGNet Training Loss"
    )

    plt.show()

    # ----------------------------------------
    # Confusion matrix
    # ----------------------------------------

    ConfusionMatrixDisplay.from_predictions(
        actual,
        predictions,
        display_labels=[
            "Left Hand",
            "Right Hand"
        ]
    )

    plt.title(
        "EEGNet Motor Imagery Classification"
    )

    plt.show()


if __name__ == "__main__":
    main()