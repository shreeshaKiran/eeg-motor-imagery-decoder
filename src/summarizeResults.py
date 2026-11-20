"""Paired statistics and figures for the CSP-LDA 2x2 (full vs C3/Cz/C4, raw vs aligned).

Reads results/tables/csp_lda_{raw,ea}_{full,C3-Cz-C4}_per_subject.csv (written by run_loso).
Pre-specified primary comparison: full-channel raw vs full-channel aligned.
The other three pairs are secondary and Holm-corrected across those three.
Writes results/tables/paired_comparisons.csv, results/tables/condition_summary.csv and
results/figures/paired_balanced_accuracy.png.

Exploratory (full run, 2026-11-08): EEGNet 64 ch raw/aligned and CSP-LDA motor strip (21 ch)
raw/aligned. These comparisons were specified AFTER the results existed, so they are exploratory,
form their own Holm family (column p_holm_secondary in the exploratory table), and use a fresh
bootstrap RNG (seed 42) so the CSP 2x2 numbers above are unaffected. Writes
results/tables/paired_comparisons_exploratory.csv and results/tables/condition_summary_exploratory.csv.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from main import MOTOR_STRIP

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

STRIP_TAG = "-".join(MOTOR_STRIP)
EXPL_CONDS = {
    "eegnet_full_raw": "eegnet_raw_full",
    "eegnet_full_ea": "eegnet_ea_full",
    "strip_raw": f"{NAME}_raw_{STRIP_TAG}",
    "strip_ea": f"{NAME}_ea_{STRIP_TAG}",
    # EEGNet channel subsets (requested 2026-11-19, after the 64-ch results; exploratory)
    "eegnet_3ch_raw": "eegnet_raw_C3-Cz-C4",
    "eegnet_3ch_ea": "eegnet_ea_C3-Cz-C4",
    "eegnet_strip_raw": f"eegnet_raw_{STRIP_TAG}",
    "eegnet_strip_ea": f"eegnet_ea_{STRIP_TAG}",
}
# conditions only needed as comparison partners (already in condition_summary.csv)
EXPL_REFS = {"full_ea": CONDS["full_ea"]}
EXPL_PAIRS = [
    ("[exploratory] alignment effect, EEGNet 64 ch", "eegnet_full_raw", "eegnet_full_ea", False),
    ("[exploratory] alignment effect, CSP-LDA motor strip 21 ch", "strip_raw", "strip_ea", False),
    ("[exploratory] CSP-LDA vs EEGNet, 64 ch aligned", "eegnet_full_ea", "full_ea", False),
    # appended 2026-11-19 (EEGNet subsets); same Holm family, so earlier Holm values change
    ("[exploratory] alignment effect, EEGNet C3/Cz/C4", "eegnet_3ch_raw", "eegnet_3ch_ea", False),
    ("[exploratory] alignment effect, EEGNet motor strip 21 ch", "eegnet_strip_raw", "eegnet_strip_ea", False),
    ("[exploratory] EEGNet channels (3 vs 64), aligned", "eegnet_full_ea", "eegnet_3ch_ea", False),
    ("[exploratory] EEGNet channels (21 vs 64), aligned", "eegnet_full_ea", "eegnet_strip_ea", False),
]

# every offline LOSO condition, for the per-subject binomial test
ALL_TAGS = {**{f"csp_lda_{k}": t for k, t in CONDS.items()},
            "csp_lda_strip_raw": EXPL_CONDS["strip_raw"], "csp_lda_strip_ea": EXPL_CONDS["strip_ea"],
            **{k: t for k, t in EXPL_CONDS.items() if k.startswith("eegnet_")}}
ALPHA = 0.05


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


def summarize(data):
    rows = []
    for k, df in data.items():
        b = df.balanced_accuracy
        rows.append({"condition": k, "n_subjects": len(df), "mean_bal_acc": b.mean(), "sd": b.std(),
                     "median": b.median(), "min": b.min(), "max": b.max(),
                     "n_ge_0.70": int((b >= 0.70).sum()), "n_lt_0.55": int((b < 0.55).sum()),
                     "n_single_class": int(df.single_class.sum()),
                     "mean_macro_f1": df.macro_f1.mean()})
    return pd.DataFrame(rows)


def compare(data, pairs, rng):
    """Paired B - A balanced accuracy; Holm over the non-primary pairs of this family."""
    out = []
    for label, a, b, primary in pairs:
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
    return comp


def main():
    data = {k: load(t) for k, t in CONDS.items()}
    rng = np.random.default_rng(SEED)

    # per-condition summary
    summ = summarize(data)

    # paired comparisons
    comp = compare(data, PAIRS, rng)

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

    exploratory()
    above_chance()
    all_conditions_figure()


def exploratory():
    """Full-run (2026-11-08) comparisons, specified after results existed: own Holm family."""
    data = {k: load(t) for k, t in EXPL_CONDS.items()}
    refs = {k: load(t) for k, t in EXPL_REFS.items()}
    rng = np.random.default_rng(SEED)
    summ = summarize(data)
    comp = compare({**data, **refs}, EXPL_PAIRS, rng)
    summ.to_csv(os.path.join(RESULTS, "tables", "condition_summary_exploratory.csv"), index=False)
    comp.to_csv(os.path.join(RESULTS, "tables", "paired_comparisons_exploratory.csv"), index=False)
    print("\n" + "=" * 60 + "\nEXPLORATORY (specified after results; separate Holm family)\n")
    print(summ.round(3).to_string(index=False))
    print()
    print(comp.round(4).to_string(index=False))


def above_chance():
    """Per-subject one-sided binomial test vs chance (p = 0.5), alpha 0.05, NO multiple-comparison
    correction. Counts correct trials from each subject's saved predictions
    (results/predictions/<tag>/S*.npz). This uses ACCURACY on roughly balanced classes
    (~22-23 trials per class per subject), not balanced accuracy.
    Writes results/tables/above_chance_counts.csv."""
    rows = []
    for cond, tag in ALL_TAGS.items():
        pdir = os.path.join(RESULTS, "predictions", tag)
        files = sorted(f for f in os.listdir(pdir) if f.startswith("S") and f.endswith(".npz"))
        n_sig, ks, ns = 0, [], []
        for f in files:
            d = np.load(os.path.join(pdir, f), allow_pickle=True)
            k, n = int((d["y_true"] == d["y_pred"]).sum()), len(d["y_true"])
            n_sig += binomtest(k, n, 0.5, alternative="greater").pvalue < ALPHA
            ks.append(k); ns.append(n)
        rows.append({"condition": cond, "tag": tag, "n_subjects": len(files),
                     "n_above_chance": int(n_sig), "frac_above_chance": n_sig / len(files),
                     "median_trials_per_subject": float(np.median(ns)),
                     "mean_accuracy": float(np.mean(np.array(ks) / np.array(ns)))})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "tables", "above_chance_counts.csv"), index=False)
    print("\n" + "=" * 60 + "\nPER-SUBJECT BINOMIAL TEST vs 0.5 (one-sided, alpha 0.05, uncorrected; accuracy)\n")
    print(out.drop(columns="tag").round(3).to_string(index=False))


def all_conditions_figure():
    """Grouped bars for all 12 LOSO conditions (model x channels x alignment): mean balanced
    accuracy, per-subject points, chance line. Reads only the per-subject CSVs.
    Writes results/figures/all_conditions.png (no other output)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    colors = {"raw": "#2a78d6", "ea": "#eb6834"}          # categorical slots 1, 2
    models = [("CSP-LDA", "csp_lda"), ("EEGNet", "eegnet")]
    chans = [("64 ch", "full"), ("21 ch (motor strip)", STRIP_TAG), ("3 ch (C3/Cz/C4)", "C3-Cz-C4")]
    rng = np.random.default_rng(SEED)                    # jitter only
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, (mlabel, mtag) in zip(axes, models):
        for i, (clabel, ctag) in enumerate(chans):
            for j, ver in enumerate(("raw", "ea")):
                b = load(f"{mtag}_{ver}_{ctag}").balanced_accuracy.to_numpy()
                x = i + (j - 0.5) * 0.38
                ax.bar(x, b.mean(), width=0.34, color=colors[ver], alpha=0.35,
                       edgecolor=colors[ver], linewidth=1.2, zorder=1)
                ax.scatter(x + rng.uniform(-0.1, 0.1, len(b)), b, s=9, color=colors[ver],
                           edgecolor="white", linewidth=0.4, zorder=3)
                ax.text(x, 1.02, f"{b.mean():.3f}", ha="center", va="bottom", fontsize=7.5, color="#333")
        ax.axhline(0.5, ls="--", color="#444", lw=0.8, zorder=2)
        ax.set_xticks(range(len(chans)))
        ax.set_xticklabels([c for c, _ in chans], fontsize=8.5)
        ax.set_title(mlabel, fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis="y", color="#ddd", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("balanced accuracy (LOSO, 40 subjects)")
    handles = [Patch(facecolor=colors["raw"], alpha=.6, label="raw"),
               Patch(facecolor=colors["ea"], alpha=.6, label="Euclidean-aligned"),
               Line2D([], [], ls="--", color="#444", lw=.8, label="chance (0.5)")]
    fig.legend(handles=handles, fontsize=8, loc="upper center", ncol=3, frameon=False)
    fig.text(0.5, 0.005, "Bars: mean across subjects (value above). Points: individual held-out subjects.",
             ha="center", fontsize=7.5, color="#555")
    plt.tight_layout(rect=(0, 0.03, 1, 0.93))
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    plt.savefig(os.path.join(RESULTS, "figures", "all_conditions.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()