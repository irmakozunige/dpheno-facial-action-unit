"""
03_eda.py
---------
Exploratory Data Analysis for the schizophrenia apathy project.
Covers: missingness, apathy score distributions, AU distributions,
inter-timepoint correlations, and QC flags.

Input:  data/features.csv
Output: results/eda/  (figures + eda_report.txt)

pip install pandas numpy matplotlib seaborn scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path

DATA_CSV    = Path("data/features.csv")
OUT_DIR     = Path("results/eda")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TIMEPOINT_ORDER = ["Baseline", "FU1", "FU3"]
sns.set_theme(style="whitegrid", palette="muted")


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["timepoint"] = pd.Categorical(df["timepoint"],
                                     categories=TIMEPOINT_ORDER, ordered=True)
    return df


# ── 1. Missingness & data inventory ──────────────────────────────────────────
def report_inventory(df: pd.DataFrame) -> str:
    lines = ["=== DATA INVENTORY ===\n"]
    lines.append(f"Total rows: {len(df)}")
    lines.append(f"Patients:   {df['patient_id'].nunique()}")
    lines.append(f"\nTimepoint counts:")
    lines.append(df["timepoint"].value_counts().sort_index().to_string())

    # which patients have which timepoints
    pivot = df.groupby(["patient_id", "timepoint"]).size().unstack(fill_value=0)
    lines.append(f"\nPer-patient timepoint matrix:")
    lines.append(pivot.to_string())

    # completers
    has_baseline = set(df[df.timepoint == "Baseline"]["patient_id"])
    has_fu3      = set(df[df.timepoint == "FU3"]["patient_id"])
    completers   = has_baseline & has_fu3
    lines.append(f"\nPatients with both Baseline + FU3: {len(completers)} → {sorted(completers)}")

    # missing apathy score
    miss = df["apathy_score"].isna().sum()
    lines.append(f"\nMissing apathy scores: {miss}/{len(df)}")

    # feature missingness
    feat_cols = [c for c in df.columns
                 if c not in ("patient_id", "timepoint", "apathy_score")]
    miss_pct = df[feat_cols].isna().mean().sort_values(ascending=False)
    high_miss = miss_pct[miss_pct > 0.1]
    if not high_miss.empty:
        lines.append(f"\nFeature columns with >10% missingness:")
        lines.append(high_miss.to_string())

    return "\n".join(lines)


# ── 2. Apathy score distributions ────────────────────────────────────────────
def plot_apathy_distributions(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle("Apathy Score Distribution by Timepoint", fontsize=13)

    for ax, tp in zip(axes, TIMEPOINT_ORDER):
        sub = df[df.timepoint == tp]["apathy_score"].dropna()
        if sub.empty:
            ax.set_title(f"{tp} (no data)")
            continue
        ax.hist(sub, bins=min(10, len(sub)), edgecolor="white", color="#4878D0")
        ax.axvline(sub.mean(), color="red", ls="--", label=f"Mean={sub.mean():.1f}")
        ax.set_title(f"{tp}  (n={len(sub)})")
        ax.set_xlabel("Apathy Score")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "apathy_distributions.png", dpi=150)
    plt.close()


def plot_apathy_trajectories(df: pd.DataFrame):
    """Spaghetti plot: each patient's apathy score over time."""
    fig, ax = plt.subplots(figsize=(8, 5))

    tp_map = {t: i for i, t in enumerate(TIMEPOINT_ORDER)}
    for pid, grp in df.dropna(subset=["apathy_score"]).groupby("patient_id"):
        grp_sorted = grp.sort_values("timepoint")
        x = [tp_map[t] for t in grp_sorted["timepoint"] if t in tp_map]
        y = grp_sorted["apathy_score"].values
        ax.plot(x, y, "o-", alpha=0.5, lw=1.2, markersize=4)
        ax.text(x[-1] + 0.05, y[-1], pid, fontsize=7, va="center")

    # group mean
    means = df.dropna(subset=["apathy_score"]).groupby("timepoint")["apathy_score"].mean()
    ax.plot([tp_map[t] for t in means.index if t in tp_map],
            means.values, "k--o", lw=2.5, markersize=8, label="Group mean", zorder=5)

    ax.set_xticks(range(len(TIMEPOINT_ORDER)))
    ax.set_xticklabels(TIMEPOINT_ORDER)
    ax.set_ylabel("Apathy Score")
    ax.set_title("Individual Apathy Trajectories")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "apathy_trajectories.png", dpi=150)
    plt.close()


# ── 3. AU feature distributions ──────────────────────────────────────────────
def plot_au_heatmap(df: pd.DataFrame):
    """Heatmap of mean AU intensities per timepoint."""
    au_cols = [c for c in df.columns if c.endswith("_mean") and "AU" in c]
    if not au_cols:
        return

    pivot = df.groupby("timepoint")[au_cols].mean()
    pivot.columns = [c.replace("_mean", "") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(max(12, len(au_cols) * 0.6), 4))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", ax=ax,
                linewidths=0.3, cbar_kws={"label": "Mean Intensity"})
    ax.set_title("Mean AU Intensities by Timepoint")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "au_heatmap.png", dpi=150)
    plt.close()


def plot_au_apathy_correlations(df: pd.DataFrame):
    """Bar chart of Spearman r between each AU feature and apathy score."""
    sub = df.dropna(subset=["apathy_score"])
    feat_cols = [c for c in sub.columns
                 if c not in ("patient_id", "timepoint", "apathy_score")
                 and sub[c].notna().sum() > 5]

    corrs = {}
    for col in feat_cols:
        try:
            r, p = stats.spearmanr(sub[col].dropna(),
                                   sub.loc[sub[col].notna(), "apathy_score"])
            corrs[col] = (r, p)
        except Exception:
            pass

    if not corrs:
        return

    corr_df = (pd.DataFrame(corrs, index=["r", "p"])
                 .T.sort_values("r"))
    # keep top/bottom 20
    n_show = min(20, len(corr_df))
    top = pd.concat([corr_df.head(n_show // 2), corr_df.tail(n_show // 2)])

    fig, ax = plt.subplots(figsize=(8, max(5, n_show * 0.35)))
    colors = ["#D65F5F" if r < 0 else "#4878D0" for r in top["r"]]
    ax.barh(top.index, top["r"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Spearman r  (with apathy score)")
    ax.set_title("Feature–Apathy Correlations (top/bottom)")
    # mark significant (p < 0.05) — cautious given small N
    for i, (name, row) in enumerate(top.iterrows()):
        if row["p"] < 0.05:
            ax.text(row["r"] + 0.01 * np.sign(row["r"]), i, "*", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "au_apathy_correlations.png", dpi=150)
    plt.close()

    return corr_df


# ── 4. QC: OpenFace confidence ────────────────────────────────────────────────
def plot_confidence_qc(df: pd.DataFrame):
    if "mean_confidence" not in df.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    for tp, grp in df.groupby("timepoint"):
        ax.scatter(grp["patient_id"], grp["mean_confidence"],
                   label=str(tp), alpha=0.7, s=60)
    ax.axhline(0.8, color="red", ls="--", label="Threshold (0.8)")
    ax.set_xlabel("Patient ID")
    ax.set_ylabel("Mean OpenFace Confidence")
    ax.set_title("OpenFace Tracking Confidence QC")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "confidence_qc.png", dpi=150)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load()

    print("=== Running EDA ===")
    report = report_inventory(df)
    print(report)
    with open(OUT_DIR / "eda_report.txt", "w") as f:
        f.write(report)

    plot_apathy_distributions(df)
    plot_apathy_trajectories(df)
    plot_au_heatmap(df)
    corr_df = plot_au_apathy_correlations(df)
    plot_confidence_qc(df)

    if corr_df is not None:
        corr_df.to_csv(OUT_DIR / "au_apathy_correlations.csv")
        print(f"\nTop 10 features correlated with apathy:")
        print(corr_df.tail(10)[["r", "p"]].to_string())

    print(f"\n✓ EDA complete. Figures saved to {OUT_DIR}/")
