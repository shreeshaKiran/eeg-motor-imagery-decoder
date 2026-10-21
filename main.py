import argparse
import os

import pandas as pd

from src.preprocessing import load_dataset
from src.subject_independent import run_loso
from src.csp_baseline import csp_lda_fit_predict
from src.utils import set_seed, log_versions

MOTOR_CHANNELS = ["C3", "Cz", "C4"]


def run_csp_lda_2x2(X, y, groups, channel_names, results_dir="results"):
    """CSP-LDA LOSO: (full 64 ch vs C3/Cz/C4) x (raw vs Euclidean-aligned)."""
    print("\n" + "=" * 60)
    print("CSP-LDA 2x2 (channels x alignment)")
    print("=" * 60)

    summary_rows = []
    for channels in (None, MOTOR_CHANNELS):
        for align in (False, True):
            df = run_loso("csp_lda", csp_lda_fit_predict, X, y, groups, channel_names,
                          align=align, channels=channels, out_dir=results_dir)
            row = {"model": "CSP-LDA",
                   "channels": "full_64" if channels is None else "-".join(channels),
                   "aligned": align, "n_subjects": len(df)}
            for m in ("accuracy", "macro_f1", "balanced_accuracy"):
                row[f"{m}_mean"] = df[m].mean()
                row[f"{m}_std"] = df[m].std()      # sample std (ddof=1), as in run_loso
            summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    path = os.path.join(results_dir, "tables", "csp_lda_2x2_summary.csv")
    summary.to_csv(path, index=False)
    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nSaved summary to {path}")
    return summary


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--csp-only", action="store_true",
                        help="run only the CSP-LDA 2x2 (skip the KNN/EEGNet electrode analysis)")
    args = parser.parse_args()

    set_seed(42)
    log_versions()

    print("=" * 60)
    print("EEG MOTOR IMAGERY NEURAL DECODER")
    print("=" * 60)

    # Load and preprocess EEG
    X, y, groups, channel_names = load_dataset()

    print("\nDataset loaded.")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Number of subjects:", len(set(groups)))
    print("Number of channels:", len(channel_names))

    # CSP-LDA 2x2 through the shared LOSO runner
    run_csp_lda_2x2(X, y, groups, channel_names, results_dir="results")

    if args.csp_only:
        return

    # Imported here: electrode_analysis still imports run_subject_independent, which the
    # new src/subject_independent.py no longer defines (see audit). Keeps --csp-only usable.
    from src.electrode_analysis import run_electrode_analysis

    # Run all model/channel comparisons
    results = run_electrode_analysis(
        X,
        y,
        groups,
        channel_names,
        results_dir="results",
    )

    print("\nProject complete.")
    print("Results saved in results/electrode_comparison.csv")


if __name__ == "__main__":
    main()