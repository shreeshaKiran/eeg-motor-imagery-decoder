"""Loading, filtering, epoching and labelling for the PhysioNet EEG Motor Movement/Imagery dataset.

load_dataset() returns (X, y, groups, channel_names):
    X              float32, shape (trials, channels, time), in VOLTS
    y              int, 0 = left-fist imagery (T1), 1 = right-fist imagery (T2)
    groups         int, subject ID per trial (use for leave-one-subject-out splits)
    channel_names  list[str]
"""
from pathlib import Path

import numpy as np
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf

# ---------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]      # folder that contains src/
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"            # EDF downloads go here
CACHE_PATH = PROJECT_ROOT / "data" / "cache" / "mi_subjects_1_40.npz"   # delete after changing any setting below

# Accept plain strings too (e.g. RAW_DATA_DIR = "/Users/you/eeg_data")
RAW_DATA_DIR = Path(RAW_DATA_DIR)
CACHE_PATH = Path(CACHE_PATH)

# ---------------------------------------------------------------- configuration
SUBJECTS = list(range(1, 41))      # subjects 1-40 (avoids known problematic IDs 88, 89, 92, 100, 104, 106)
RUNS = [4, 8, 12]                  # imagined left- vs right-fist movement

LOW_FREQ = 8                       # Hz, bandpass low cut (mu band starts ~8 Hz)
HIGH_FREQ = 30                     # Hz, bandpass high cut (beta band ends ~30 Hz)

EPOCH_START = 1.0                  # s after cue onset
EPOCH_END = 4.0                    # s after cue onset

EXPECTED_FS = 160                  # Hz

LABEL_MAP = {"left": 0, "right": 1}


# ---------------------------------------------------------------- per-subject steps
def load_subject(subject):
    """Load, filter (per run) and concatenate the imagery runs for one subject."""
    print(f"\nLoading subject {subject}")

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_paths = eegbci.load_data(subjects=subject, runs=RUNS, path=str(RAW_DATA_DIR))

    raws = []
    for file_path in file_paths:
        raw = read_raw_edf(file_path, preload=True, verbose=False)
        # Filter each run separately so the filter does not ring across run boundaries
        raw.filter(LOW_FREQ, HIGH_FREQ, verbose=False)
        raws.append(raw)

    raw = mne.concatenate_raws(raws)
    eegbci.standardize(raw)          # clean channel names ('C3.' -> 'C3') and set montage
    return raw


def preprocess(raw):
    """Keep EEG channels and apply an average reference."""
    raw.pick("eeg")
    raw.set_eeg_reference("average", verbose=False)
    return raw


def create_epochs(raw):
    """Epoch around T1 (left) and T2 (right) annotations, mapped by description."""
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

    selected_events = {}
    if "T1" in event_dict:
        selected_events["left"] = event_dict["T1"]
    if "T2" in event_dict:
        selected_events["right"] = event_dict["T2"]
    if len(selected_events) != 2:
        raise ValueError("Could not find both T1 and T2 events.")

    epochs = mne.Epochs(
        raw,
        events,
        event_id=selected_events,
        tmin=EPOCH_START,
        tmax=EPOCH_END,
        baseline=None,
        reject_by_annotation=True,   # drop epochs overlapping run-boundary (BAD/EDGE) annotations
        preload=True,
        verbose=False,
    )
    return epochs


# ---------------------------------------------------------------- dataset
def load_dataset(use_cache=True):
    """Build (or load from cache) the full multi-subject dataset."""
    if use_cache and CACHE_PATH.exists():
        d = np.load(CACHE_PATH, allow_pickle=True)
        return d["X"], d["y"], d["groups"], list(d["channel_names"])

    all_X, all_y, all_groups = [], [], []
    channel_names = None
    skipped = []

    for subject in SUBJECTS:
        try:
            raw = preprocess(load_subject(subject))

            if raw.info["sfreq"] != EXPECTED_FS:
                raise ValueError(f"unexpected sampling rate {raw.info['sfreq']}")

            epochs = create_epochs(raw)

            if channel_names is None:
                channel_names = list(epochs.ch_names)
            elif list(epochs.ch_names) != channel_names:
                raise ValueError("channel names differ from earlier subjects")

        except Exception as e:
            print(f"SKIPPING subject {subject}: {e}")
            skipped.append(subject)
            continue

        X_subject = epochs.get_data().astype(np.float32)      # volts
        event_ids = epochs.events[:, -1]
        y_subject = (event_ids == epochs.event_id["right"]).astype(int)

        all_X.append(X_subject)
        all_y.append(y_subject)
        all_groups.extend([subject] * len(y_subject))

        print(f"Subject {subject}: {len(y_subject)} trials, "
              f"class counts (left, right) = {np.bincount(y_subject, minlength=2)}")

    if not all_X:
        raise RuntimeError("No subjects could be loaded.")

    X = np.concatenate(all_X)
    y = np.concatenate(all_y)
    groups = np.array(all_groups)

    print(f"\nLoaded {len(np.unique(groups))} subjects, {len(y)} trials total.")
    print("Skipped subjects:", skipped if skipped else "none")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH, X=X, y=y, groups=groups, channel_names=np.array(channel_names)
    )
    return X, y, groups, channel_names


if __name__ == "__main__":
    X, y, groups, ch = load_dataset(use_cache=False)
    print(X.shape, y.shape, groups.shape, len(ch))
