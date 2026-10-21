"""Paired statistics and figures for the CSP-LDA 2x2 (full vs C3/Cz/C4, raw vs aligned).

Reads results/tables/csp_lda_{raw,ea}_{full,C3-Cz-C4}_per_subject.csv (written by run_loso).
Pre-specified primary comparison: full-channel raw vs full-channel aligned.
The other three pairs are secondary and Holm-corrected across those three.
Writes results/tables/paired_comparisons.csv, results/tables/condition_summary.csv and
results/figures/paired_balanced_accuracy.png.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RESULTS = "results"
NAME = "csp_lda"
CONDS = {
    "full_raw": f"{NAME}_raw_full",
    "full_ea": f"{NAME}_ea_full",
    "3ch_raw": f"{NAME}_raw_C3-Cz-C4",
    "3ch_ea": f"{NAME}_ea_C3-Cz-C4",
}
# (label, condition A, condition B, primary?)  -> difference = B - A
PAIRS = [
    ("alignment effect, full 64 ch", "full_raw", "full_ea", True),
    ("alignment effect, C3/Cz/C4", "3ch_raw", "3ch_ea", False),
    ("channels (3 vs 64), raw", "full_raw", "3ch_raw", False),
    ("channels (3 vs 64), aligned", "full_ea", "3ch_ea", False),
]
SEED, N_BOOT = 42, 10000


def load(tag):
    df = pd.read_csv(os.path.join(RESULTS, "tables", f"{tag}_per_subject.csv")).set_index("subject")
    df["single_class"] = ((df.fp + df.tp) == 0) | ((df.tn + df.fn) == 0)
    return df


def holm(pvals):
    order = np.argsort(pvals)
    adj = np.empty(len(pvals))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvals) - rank) * pvals[i])
        adj[i] = min(1.0, running)
    return adj


def main():
    data = {k: load(t) for k, t in CONDS.items()}
    rng = np.random.default_rng(SEED)

    # per-condition summary
    rows = []
    for k, df in data.items():
        b = df.balanced_accuracy
        rows.append({"condition": k, "n_subjects": len(df), "mean_bal_acc": b.mean(), "sd": b.std(),
                     "median": b.median(), "min": b.min(), "max": b.max(),
                     "n_ge_0.70": int((b >= 0.70).sum()), "n_lt_0.55": int((b < 0.55).sum()),
                     "n_single_class": int(df.single_class.sum()),
                     "mean_macro_f1": df.macro_f1.mean()})
    summ = pd.DataFrame(rows)

    # paired comparisons
    out = []
    for label, a, b, primary in PAIRS:
        A, B = data[a].balanced_accuracy, data[b].balanced_accuracy
        idx = A.index.intersection(B.index)
        d = (B.loc[idx] - A.loc[idx]).to_numpy()
        boots = rng.choice(d, size=(N_BOOT, len(d)), replace=True).mean(1)
        try:
            p = wilcoxon(d, zero_method="wilcox").pvalue
        except ValueError:
            p = 1.0
        out.append({"comparison": label, "B_minus_A": f"{b} - {a}", "primary": primary,
                    "n_subjects": len(d), "mean_diff": d.mean(), "median_diff": float(np.median(d)),
                    "ci95_low": np.percentile(boots, 2.5), "ci95_high": np.percentile(boots, 97.5),
                    "n_improved": int((d > 0).sum()), "n_worse": int((d < 0).sum()), "n_equal": int((d == 0).sum()),
                    "wilcoxon_p": p})
    comp = pd.DataFrame(out)
    sec = ~comp.primary.to_numpy()
    comp["p_holm_secondary"] = np.nan
    comp.loc[sec, "p_holm_secondary"] = holm(comp.loc[sec, "wilcoxon_p"].to_numpy())

    os.makedirs(os.path.join(RESULTS, "tables"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    summ.to_csv(os.path.join(RESULTS, "tables", "condition_summary.csv"), index=False)
    comp.to_csv(os.path.join(RESULTS, "tables", "paired_comparisons.csv"), index=False)
    pd.set_option("display.width", 200)
    print(summ.round(3).to_string(index=False))
    print()
    print(comp.round(4).to_string(index=False))

    # paired figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, (a, b, title) in zip(axes, [("full_raw", "full_ea", "64 channels"), ("3ch_raw", "3ch_ea", "C3/Cz/C4")]):
        A, B = data[a].balanced_accuracy, data[b].balanced_accuracy
        for s in A.index:
            ax.plot([0, 1], [A[s], B[s]], color="gray", alpha=.4, lw=.8)
        ax.plot([0, 1], [A.mean(), B.mean()], color="C3", lw=3, marker="o")
        ax.axhline(0.5, ls="--", color="k", lw=.8)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["raw", "aligned"]); ax.set_title(title)
    axes[0].set_ylabel("balanced accuracy (LOSO)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS, "figures", "paired_balanced_accuracy.png"), dpi=200)


if __name__ == "__main__":
    main()