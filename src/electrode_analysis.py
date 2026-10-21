import os
import pandas as pd

from .knn_baseline import run_knn
from .subject_independent import run_subject_independent


def select_channels(X, channel_names, selected_channels):

    missing = [
        ch for ch in selected_channels
        if ch not in channel_names
    ]

    if missing:
        raise ValueError(f"Missing channels: {missing}")

    indices = [
        channel_names.index(ch)
        for ch in selected_channels
    ]

    return X[:, indices, :]


def get_channel_sets(channel_names):

    motor_channels = ["C3", "Cz", "C4"]

    if not all(ch in channel_names for ch in motor_channels):
        raise ValueError(
            "C3, Cz, and C4 must all be present."
        )

    return {
        "all_channels": list(channel_names),
        "motor_3": motor_channels,
    }


def run_electrode_analysis(
    X,
    y,
    groups,
    channel_names,
    results_dir="results",
):

    print("\n" + "=" * 60)
    print("ELECTRODE REDUCTION ANALYSIS")
    print("=" * 60)

    os.makedirs(results_dir, exist_ok=True)

    channel_sets = get_channel_sets(channel_names)

    all_results = []

    for set_name, selected_channels in channel_sets.items():

        print(f"\nChannel configuration: {set_name}")
        print(f"Number of channels: {len(selected_channels)}")

        X_subset = select_channels(
            X,
            channel_names,
            selected_channels,
        )

        # Each model is evaluated once for this channel set.

        knn_output = run_knn(
            X_subset, y, groups
        )

        eegnet_output = run_subject_independent(
            X_subset, y, groups
        )

        for result in knn_output["subject_results"]:
            all_results.append({
                "model": "KNN",
                "channel_set": set_name,
                "n_channels": len(selected_channels),
                **result,
            })

        for result in eegnet_output:
            all_results.append({
                "model": "EEGNet",
                "channel_set": set_name,
                "n_channels": len(selected_channels),
                **result,
            })

    results = pd.DataFrame(all_results)

    output_path = os.path.join(
        results_dir,
        "electrode_comparison.csv",
    )

    results.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("ELECTRODE COMPARISON SUMMARY")
    print("=" * 60)

    summary = (
        results.groupby(["model", "channel_set", "n_channels"])
        [["accuracy", "macro_f1", "balanced_accuracy"]]
        .agg(["mean", "std"])
        .round(4)
    )

    print(summary)
    print(f"\nSaved subject-level results to: {output_path}")

    return results