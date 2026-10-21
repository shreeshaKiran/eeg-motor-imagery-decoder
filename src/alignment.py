"""Euclidean alignment (He & Wu, 2020) of EEG trials, done independently per subject.

For each subject, R = mean over trials of (X X^T / T) and X_aligned = R^{-1/2} X.
This uses ONLY that subject's unlabeled trials (no labels, no other subjects), so it can be
applied once up front to every subject, including the held-out one. It is transductive in the
unlabeled sense: the held-out subject's own trials are used to estimate their R. State this
in the README.

Average-referenced EEG has rank (n_channels - 1), so R is singular; a pseudo-inverse square
root (dropping near-zero eigenvalues) is used.
"""
import numpy as np


def euclidean_align(X, tol=1e-10):
    """X: (trials, channels, time) -> aligned array, same shape (float32)."""
    X64 = X.astype(np.float64)
    T = X64.shape[2]
    R = np.einsum("nct,ndt->cd", X64, X64) / (T * X64.shape[0])
    vals, vecs = np.linalg.eigh(R)
    keep = vals > tol * vals.max()
    V = vecs[:, keep]
    inv_sqrt = (V / np.sqrt(vals[keep])) @ V.T
    return np.einsum("cd,ndt->nct", inv_sqrt, X64).astype(np.float32)


def align_by_subject(X, groups):
    """Apply euclidean_align separately to each subject's trials."""
    X_out = np.empty_like(X, dtype=np.float32)
    for s in np.unique(groups):
        m = groups == s
        X_out[m] = euclidean_align(X[m])
    return X_out
