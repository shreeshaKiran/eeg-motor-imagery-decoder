"""SIMULATED real-time decoding demo on PRE-RECORDED PhysioNet data (CSP-LDA, 64 channels).

This is a demo, separate from the offline LOSO results. It is NOT a working BCI: no live
data, no user feedback, no closed loop. The design below was fixed before running.

Model: CSP-LDA trained per held-out subject on the other 39 subjects. Number of CSP components
is chosen by the inner GroupKFold of src.csp_baseline.csp_lda_fit_predict (training subjects
only); the final pipeline is then refit with that number (deterministic, same as inside).

Preprocessing (training AND streaming): 4th-order Butterworth 8-30 Hz bandpass as SOS, applied
CAUSALLY with scipy.signal.sosfilt (filter starts from zero state at the start of each run; the
state is carried across chunks while streaming), then average reference per sample. This differs
ON PURPOSE from the offline pipeline (MNE zero-phase FIR, which needs future samples). Training
epochs are cue+1 s to cue+4 s (480 samples, end exclusive), cut from these causally filtered runs,
not from the offline cache.

Versions:
  raw      no alignment.
  aligned  training subjects: Euclidean alignment per subject on all their epochs (as offline).
           held-out subject: R estimated ONLY from the unlabeled epochs of run 4; that fixed
           R^(-1/2) is applied to every streamed window of runs 8 and 12.
Both versions are evaluated only on runs 8 and 12 (same trials).

Streaming: runs 8 and 12 are fed in 0.5 s chunks (80 samples). After each chunk the latest 3 s
window (480 samples) gives P(right) and a decision. Built-in delay: every decision uses the last
3 s of signal, so it reflects activity from up to 3 s in the past.

Evaluation timing: decisions fall on an 80-sample grid that is not aligned to the cues, so the
decision for "window ending tau s after cue" is the one whose window end is NEAREST to cue + tau
(|offset| <= 0.25 s; the mean absolute offset is reported). tau = 4.0 s is the trial decision
(the offline epoch window, up to that offset); tau = 0.5 ... 4.5 s gives time-resolved balanced accuracy.
Compute time per decision = filter + average reference + alignment + CSP + LDA on one chunk
(time.perf_counter).

Example subject (fixed rule, independent of demo results): lower median of the aligned offline
balanced accuracy in results/tables/csp_lda_ea_full_per_subject.csv (20th of 40 when sorted,
ties broken by lower subject ID).

Usage: python -m src.realtime_demo [--subjects 1 2]   (no --subjects = all 40)
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import mne
from mne.datasets import eegbci
from mne.io import read_raw_edf
from scipy.signal import butter, sosfilt
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from src.alignment import align_by_subject
from src.csp_baseline import csp_lda_fit_predict, make_csp_lda
from src.preprocessing import RAW_DATA_DIR, SUBJECTS, RUNS, LOW_FREQ, HIGH_FREQ, EXPECTED_FS
from src.utils import set_seed

OUT_DIR = os.path.join("results", "realtime")
SEED = 42
FS = EXPECTED_FS
CHUNK = 80                     # 0.5 s
WIN = 480                      # 3 s
EPOCH_START, EPOCH_END = 160, 640   # samples after cue (1 s, 4 s), end exclusive
ALIGN_RUN = 4
STREAM_RUNS = [8, 12]
TAUS = np.round(np.arange(0.5, 4.51, 0.5), 2)   # window end after cue, s
TRIAL_TAU = 4.0
SOS = butter(4, [LOW_FREQ, HIGH_FREQ], btype="bandpass", fs=FS, output="sos")
VERSIONS = ("raw", "aligned")


# ------------------------------------------------------------------ data
def load_runs(subject):
    """{run: (raw_volts (ch, n), cue_samples, labels)} with T1 -> 0 (left), T2 -> 1 (right)."""
    paths = eegbci.load_data(subjects=subject, runs=RUNS, path=str(RAW_DATA_DIR), verbose=False)
    out, ch_names = {}, None
    for run, p in zip(RUNS, paths):
        raw = read_raw_edf(p, preload=True, verbose=False)
        eegbci.standardize(raw)
        raw.pick("eeg")
        if raw.info["sfreq"] != FS:
            raise ValueError(f"S{subject} run {run}: sfreq {raw.info['sfreq']}")
        if ch_names is None:
            ch_names = list(raw.ch_names)
        events, ev = mne.events_from_annotations(raw, verbose=False)
        keep = np.isin(events[:, 2], [ev["T1"], ev["T2"]])
        cues = events[keep, 0] - raw.first_samp
        labels = (events[keep, 2] == ev["T2"]).astype(int)
        out[run] = (raw.get_data().astype(np.float64), cues, labels)
    return out, ch_names


def causal_filter(x, zi=None):
    """Causal SOS bandpass along time + average reference. Returns (y, new_state)."""
    if zi is None:
        zi = np.zeros((SOS.shape[0], x.shape[0], 2))
    y, zf = sosfilt(SOS, x, axis=1, zi=zi)
    return y - y.mean(axis=0, keepdims=True), zf


def epochs_from_run(x_filt, cues, labels):
    ok = cues + EPOCH_END <= x_filt.shape[1]
    ep = np.stack([x_filt[:, c + EPOCH_START:c + EPOCH_END] for c in cues[ok]])
    return ep, labels[ok]


def inv_sqrt_ref(ep, tol=1e-10):
    """R^(-1/2) from unlabeled epochs; same math as src.alignment.euclidean_align."""
    R = np.einsum("nct,ndt->cd", ep, ep) / (ep.shape[2] * ep.shape[0])
    vals, vecs = np.linalg.eigh(R)
    keep = vals > tol * vals.max()
    V = vecs[:, keep]
    return (V / np.sqrt(vals[keep])) @ V.T


def build_training_epochs(subjects):
    """Causally filtered 1-4 s epochs (all runs) for every subject. Returns X, y, groups, runs."""
    X, y, g, r, ch_ref = [], [], [], [], None
    for s in subjects:
        runs, ch = load_runs(s)
        if ch_ref is None:
            ch_ref = ch
        elif ch != ch_ref:
            raise ValueError(f"S{s}: channel names differ")
        for run, (x, cues, labels) in runs.items():
            ep, lab = epochs_from_run(causal_filter(x)[0], cues, labels)
            X.append(ep.astype(np.float32)); y.append(lab)
            g.append(np.full(len(lab), s)); r.append(np.full(len(lab), run))
    return np.concatenate(X), np.concatenate(y), np.concatenate(g), np.concatenate(r), ch_ref


# ------------------------------------------------------------------ streaming
def stream_run(model, x_raw, W=None):
    """Feed one continuous run in CHUNK-sample chunks. Returns window-end samples, P(right), ms."""
    n_ch, n = x_raw.shape
    zi = np.zeros((SOS.shape[0], n_ch, 2))
    buf = np.zeros((n_ch, 0))
    ends, probs, times = [], [], []
    for start in range(0, n - CHUNK + 1, CHUNK):
        chunk = x_raw[:, start:start + CHUNK]
        t0 = time.perf_counter()
        y, zi = causal_filter(chunk, zi)
        buf = np.concatenate([buf, y], axis=1)[:, -WIN:]
        if buf.shape[1] < WIN:
            continue
        win = buf if W is None else W @ buf
        p = model.predict_proba(win[None])[0, 1]
        times.append((time.perf_counter() - t0) * 1e3)
        ends.append(start + CHUNK)
        probs.append(p)
    return np.array(ends), np.array(probs), np.array(times)


def decisions_at(ends, probs, cues, tau):
    """P(right) of the decision whose window end is nearest cue + tau s (NaN if none within 0.25 s)."""
    target = cues + int(round(tau * FS))
    out, offs = np.full(len(cues), np.nan), np.full(len(cues), np.nan)
    if len(ends) == 0:
        return out, offs
    idx = np.clip(np.searchsorted(ends, target), 0, len(ends) - 1)
    for i, t in enumerate(target):
        cand = [j for j in (idx[i] - 1, idx[i]) if 0 <= j < len(ends)]
        j = min(cand, key=lambda j: abs(ends[j] - t))
        if abs(ends[j] - t) <= CHUNK // 2:
            out[i], offs[i] = probs[j], (ends[j] - t) / FS
    return out, offs


def bal_acc(y, p):
    m = ~np.isnan(p)
    if m.sum() == 0 or len(np.unique(y[m])) < 2:
        return np.nan
    return balanced_accuracy_score(y[m], (p[m] > 0.5).astype(int))


# ------------------------------------------------------------------ per subject
def run_subject(s, train, stream_data, align):
    X, y, g, r = train
    tr = g != s
    Xtr, ytr, gtr = X[tr], y[tr], g[tr]
    W = None
    if align:
        Xtr = align_by_subject(Xtr, gtr)
        own = (g == s) & (r == ALIGN_RUN)
        W = inv_sqrt_ref(X[own].astype(np.float64))      # labels not used
    # component selection via the existing inner GroupKFold; then refit the same pipeline
    _, _, info = csp_lda_fit_predict(Xtr, ytr, gtr, Xtr[:1])
    model = make_csp_lda(info["n_components"]).fit(Xtr.astype(np.float64), ytr)

    res = {"n_components": info["n_components"], "inner_balanced_acc": info["inner_balanced_acc"]}
    all_y, all_p, all_off, all_t, streams = [], {tau: [] for tau in TAUS}, [], [], {}
    for run in STREAM_RUNS:
        x, cues, labels = stream_data[run]
        ends, probs, times = stream_run(model, x, W)
        streams[run] = {"ends": ends, "probs": probs, "cues": cues, "labels": labels}
        all_t.append(times)
        ok = cues + EPOCH_END <= x.shape[1]          # same trials as the training epochs
        all_y.append(labels[ok])
        for tau in TAUS:
            p, off = decisions_at(ends, probs, cues[ok], tau)
            all_p[tau].append(p)
            if tau == TRIAL_TAU:
                all_off.append(off)
    yv = np.concatenate(all_y)
    p4 = np.concatenate(all_p[TRIAL_TAU])
    m = ~np.isnan(p4)
    t = np.concatenate(all_t)
    res.update({"n_trials": int(m.sum()),
                "accuracy": accuracy_score(yv[m], (p4[m] > 0.5).astype(int)),
                "balanced_accuracy": bal_acc(yv, p4),
                "mean_abs_offset_s": float(np.nanmean(np.abs(np.concatenate(all_off)))),
                "t_median_ms": float(np.median(t)), "t_p95_ms": float(np.percentile(t, 95)),
                "n_decisions": int(len(t))})
    for tau in TAUS:
        res[f"bal_acc_t{tau}"] = bal_acc(yv, np.concatenate(all_p[tau]))
    return res, streams


def example_subject():
    df = pd.read_csv(os.path.join("results", "tables", "csp_lda_ea_full_per_subject.csv"))
    df = df.sort_values(["balanced_accuracy", "subject"]).reset_index(drop=True)
    return int(df.subject.iloc[(len(df) - 1) // 2])   # lower median


# ------------------------------------------------------------------ outputs
def write_outputs(subjects):
    rows = []
    for s in subjects:
        for v in VERSIONS:
            path = os.path.join(OUT_DIR, "predictions", v, f"S{s}.npz")
            if os.path.exists(path):
                rows.append({"subject": s, "version": v,
                             **json.loads(str(np.load(path, allow_pickle=True)["res"]))})
    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(OUT_DIR, "per_subject.csv"), index=False)
    cols = [c for c in per.columns if c not in ("subject", "version")]
    summ = per.groupby("version")[cols].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.insert(0, "n_subjects", per.groupby("version").size())
    summ["window_delay_s"] = WIN / FS
    summ["chunk_s"] = CHUNK / FS
    summ = summ.reset_index()
    summ.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    pd.set_option("display.width", 200)
    show = ["version", "n_subjects", "accuracy_mean", "accuracy_std", "balanced_accuracy_mean",
            "balanced_accuracy_std", "mean_abs_offset_s_mean", "t_median_ms_mean", "t_p95_ms_mean",
            "window_delay_s"]
    print(summ[show].round(3).to_string(index=False))
    print("\ntime-resolved balanced accuracy (mean over subjects):")
    print(summ[["version"] + [f"bal_acc_t{t}_mean" for t in TAUS]].round(3).to_string(index=False))
    return per


def plot(per, subjects):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig_dir = os.path.join(OUT_DIR, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    for v, c in zip(VERSIONS, ("C0", "C3")):
        d = per[per.version == v]
        m = np.array([d[f"bal_acc_t{t}"].mean() for t in TAUS])
        sd = np.array([d[f"bal_acc_t{t}"].std() for t in TAUS])
        ax.plot(TAUS, m, marker="o", color=c, label=f"{v} (n={len(d)})")
        ax.fill_between(TAUS, m - sd, m + sd, color=c, alpha=.15)
    ax.axhline(0.5, ls="--", color="k", lw=.8, label="chance")
    ax.axvspan(1, 4, color="gray", alpha=.08)
    ax.set_xlabel("window end, s after cue (window = previous 3 s)")
    ax.set_ylabel("balanced accuracy (mean ± SD over subjects)")
    ax.set_title("Simulated streaming on pre-recorded data, CSP-LDA 64 ch")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "time_resolved_accuracy.png"), dpi=200)
    plt.close(fig)

    ex = example_subject()
    note = ""
    if ex not in subjects:
        note = f" (test run: rule subject S{ex} not run; showing S{subjects[0]})"
        ex = subjects[0]
    fig, axes = plt.subplots(len(STREAM_RUNS), 1, figsize=(12, 5), sharey=True)
    for ax, run in zip(axes, STREAM_RUNS):
        for v, c in zip(VERSIONS, ("C0", "C3")):
            d = np.load(os.path.join(OUT_DIR, "predictions", v, f"S{ex}.npz"), allow_pickle=True)
            ends, probs = d[f"ends_{run}"], d[f"probs_{run}"]
            ax.plot(ends / FS, probs, color=c, lw=1, label=v)
        cues, labels = d[f"cues_{run}"], d[f"labels_{run}"]
        for cu, lab in zip(cues, labels):
            ax.axvspan(cu / FS, cu / FS + 4.1, color="C1" if lab else "C2", alpha=.12, lw=0)
            ax.axvline(cu / FS, color="k", lw=.5)
        ax.axhline(0.5, ls="--", color="k", lw=.6)
        ax.set_ylabel(f"run {run}\nP(right)")
        ax.set_ylim(0, 1)
    axes[-1].set_xlabel("time in run, s (value at window end; window = previous 3 s)")
    from matplotlib.patches import Patch
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h + [Patch(color="C1", alpha=.3), Patch(color="C2", alpha=.3)],
                   l + ["true: right", "true: left"], fontsize=7, ncol=4, loc="upper right")
    fig.suptitle(f"S{ex}: simulated streaming on pre-recorded data (not a live BCI){note}", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "example_stream.png"), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, nargs="+", default=None)
    args = ap.parse_args()
    subjects = args.subjects or SUBJECTS
    set_seed(SEED)
    for v in VERSIONS:
        os.makedirs(os.path.join(OUT_DIR, "predictions", v), exist_ok=True)

    todo = [s for s in subjects
            if not all(os.path.exists(os.path.join(OUT_DIR, "predictions", v, f"S{s}.npz")) for v in VERSIONS)]
    if todo:
        print(f"Building causally filtered training epochs for {len(SUBJECTS)} subjects ...")
        X, y, g, r, _ = build_training_epochs(SUBJECTS)
        print(f"  {X.shape[0]} epochs, shape {X.shape}")
        for s in todo:
            stream_data, _ = load_runs(s)
            for v in VERSIONS:
                path = os.path.join(OUT_DIR, "predictions", v, f"S{s}.npz")
                if os.path.exists(path):
                    continue
                set_seed(SEED)
                t0 = time.time()
                res, streams = run_subject(s, (X, y, g, r), stream_data, align=(v == "aligned"))
                arrays = {f"{k}_{run}": a for run, d in streams.items() for k, a in d.items()}
                np.savez(path, res=json.dumps(res), **arrays)
                print(f"[realtime {v}] S{s}: acc={res['accuracy']:.3f} bal_acc={res['balanced_accuracy']:.3f} "
                      f"n_comp={res['n_components']} t_med={res['t_median_ms']:.2f} ms "
                      f"t_p95={res['t_p95_ms']:.2f} ms ({time.time() - t0:.0f} s)")
    per = write_outputs(subjects)
    plot(per, subjects)


if __name__ == "__main__":
    main()
