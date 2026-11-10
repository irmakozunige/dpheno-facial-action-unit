"""
04_model_shap.py
-----------------
XGBoost regression with GroupKFold CV (group = patient_id)
+ SHAP feature importance analysis.

Two experiments run back-to-back:
  A) All timepoints pooled  → "what features correlate with apathy at any stage?"
  B) Predict FU3 from Baseline features only  → "early biomarker"
     (only if enough completers exist)

Input:  data/features.csv
Output: results/model/  (metrics CSV, SHAP plots, feature rankings)

pip install xgboost shap scikit-learn pandas numpy matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import json
from pathlib import Path
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor

DATA_CSV  = Path("data/features.csv")
OUT_DIR   = Path("results/model")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS     = 5     # GroupKFold splits (≤ n_patients)
MIN_FU3_N    = 8     # minimum completers to run Experiment B

# XGBoost params tuned for small N (conservative, avoid overfitting)
XGB_PARAMS = dict(
    n_estimators    = 100,
    max_depth       = 2,       # shallow — critical for small N
    learning_rate   = 0.05,
    subsample       = 0.8,
    colsample_bytree= 0.7,
    reg_alpha       = 1.0,     # L1
    reg_lambda      = 2.0,     # L2
    random_state    = RANDOM_STATE,
    n_jobs          = -1,
)


# ── Helpers ───────────────────────────────────────────────────────────────────
def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["apathy_score"])

    # drop columns with >50% missing, then impute the rest
    feat_cols = [c for c in df.columns
                 if c not in ("patient_id", "timepoint", "apathy_score")]
    miss_rate = df[feat_cols].isna().mean()
    drop_cols = miss_rate[miss_rate > 0.5].index.tolist()
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} cols with >50% missing: {drop_cols[:5]} ...")
        df = df.drop(columns=drop_cols)

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c not in ("patient_id", "timepoint", "apathy_score")]


def evaluate(y_true, y_pred, label=""):
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)
    print(f"  [{label}]  MAE={mae:.3f}  R²={r2:.3f}  N={len(y_true)}")
    return {"label": label, "n": len(y_true), "mae": mae, "r2": r2}


def build_pipeline(model):
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale",  StandardScaler()),
        ("model",  model),
    ])


# ── SHAP analysis ─────────────────────────────────────────────────────────────
def run_shap(model, X: pd.DataFrame, feat_names: list[str], label: str):
    """
    Compute SHAP values on the full dataset (after imputation/scaling via
    the fitted pipeline). Saves beeswarm + bar plots.
    """
    # re-transform X through the pipeline's preprocessing steps
    imputer = model.named_steps["impute"]
    scaler  = model.named_steps["scale"]
    xgb     = model.named_steps["model"]

    X_proc = scaler.transform(imputer.transform(X))
    X_proc_df = pd.DataFrame(X_proc, columns=feat_names)

    explainer   = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_proc_df)

    # 1. Summary beeswarm
    fig, ax = plt.subplots(figsize=(10, min(6, len(feat_names) * 0.3 + 2)))
    shap.summary_plot(shap_values, X_proc_df, plot_type="dot",
                      show=False, max_display=20)
    plt.title(f"SHAP Summary — {label}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_beeswarm_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Bar (mean |SHAP|)
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_df   = pd.DataFrame({"feature": feat_names, "mean_abs_shap": mean_shap})
    shap_df   = shap_df.sort_values("mean_abs_shap", ascending=False)

    fig, ax = plt.subplots(figsize=(8, min(6, len(feat_names) * 0.3 + 2)))
    top = shap_df.head(20)
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="#4878D0")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Feature Importance (SHAP) — {label}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"shap_bar_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

    shap_df.to_csv(OUT_DIR / f"shap_ranking_{label}.csv", index=False)
    print(f"\n  Top 10 features by SHAP [{label}]:")
    print(shap_df.head(10).to_string(index=False))
    return shap_df


# ── Experiment A: All timepoints pooled, GroupKFold ──────────────────────────
def experiment_a(df: pd.DataFrame) -> dict:
    print("\n" + "="*60)
    print("EXPERIMENT A: All timepoints — GroupKFold by patient")
    print("="*60)

    feat_cols = get_feature_cols(df)
    # add timepoint as numeric feature
    tp_map = {"Baseline": 0, "FU1": 1, "FU3": 2}
    df = df.copy()
    df["timepoint_num"] = df["timepoint"].map(tp_map).fillna(1).astype(int)
    feat_cols = feat_cols + ["timepoint_num"]

    X      = df[feat_cols]
    y      = df["apathy_score"].values
    groups = df["patient_id"].values

    n_patients = len(np.unique(groups))
    n_splits   = min(N_SPLITS, n_patients)
    gkf        = GroupKFold(n_splits=n_splits)

    # --- XGBoost ---
    xgb_pipe = build_pipeline(XGBRegressor(**XGB_PARAMS))
    y_pred_xgb = cross_val_predict(xgb_pipe, X, y, groups=groups, cv=gkf)
    metrics_xgb = evaluate(y, y_pred_xgb, "XGBoost GroupKFold")

    # --- Ridge (more stable baseline for small N) ---
    ridge_pipe = build_pipeline(RidgeCV(alphas=[0.1, 1, 10, 100]))
    y_pred_ridge = cross_val_predict(ridge_pipe, X, y, groups=groups, cv=gkf)
    metrics_ridge = evaluate(y, y_pred_ridge, "Ridge GroupKFold")

    # Scatter: predicted vs actual
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, preds, title in zip(axes,
                                 [y_pred_xgb, y_pred_ridge],
                                 ["XGBoost", "Ridge"]):
        ax.scatter(y, preds, alpha=0.7, edgecolors="white", s=60)
        lo, hi = min(y.min(), preds.min()), max(y.max(), preds.max())
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5)
        ax.set_xlabel("Actual Apathy Score")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{title}  (R²={r2_score(y,preds):.2f})")
    fig.suptitle("Experiment A — Predicted vs Actual (GroupKFold CV)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expA_predicted_vs_actual.png", dpi=150)
    plt.close()

    # SHAP on XGBoost fitted on all data
    xgb_pipe.fit(X, y)
    shap_df = run_shap(xgb_pipe, X, feat_cols, label="ExpA")

    return {"xgb": metrics_xgb, "ridge": metrics_ridge, "shap": shap_df}


# ── Experiment B: Predict FU3 apathy from Baseline features ──────────────────
def experiment_b(df: pd.DataFrame) -> dict | None:
    print("\n" + "="*60)
    print("EXPERIMENT B: Predict FU3 from Baseline features only")
    print("="*60)

    baseline = df[df["timepoint"] == "Baseline"].copy()
    fu3      = df[df["timepoint"] == "FU3"][["patient_id", "apathy_score"]]\
                 .rename(columns={"apathy_score": "apathy_fu3"})

    merged = baseline.merge(fu3, on="patient_id", how="inner")

    if len(merged) < MIN_FU3_N:
        print(f"  ⚠ Only {len(merged)} completers (Baseline+FU3)."
              f"  Need ≥{MIN_FU3_N} — skipping Experiment B.")
        return None

    print(f"  Completers (Baseline + FU3): {len(merged)}")

    feat_cols = get_feature_cols(merged)
    # remove FU3 apathy from features if present
    feat_cols = [c for c in feat_cols if "apathy" not in c.lower()]

    X = merged[feat_cols]
    y = merged["apathy_fu3"].values

    # leave-one-patient-out (best for tiny N)
    from sklearn.model_selection import LeaveOneGroupOut
    logo   = LeaveOneGroupOut()
    groups = merged["patient_id"].values

    xgb_pipe = build_pipeline(XGBRegressor(**XGB_PARAMS))
    y_pred   = cross_val_predict(xgb_pipe, X, y, groups=groups, cv=logo)
    metrics  = evaluate(y, y_pred, "XGBoost LOGO-CV (Baseline→FU3)")

    # scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y, y_pred, alpha=0.8, s=80, edgecolors="white")
    lo, hi = min(y.min(), y_pred.min()), max(y.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], "r--")
    ax.set_xlabel("Actual FU3 Apathy Score")
    ax.set_ylabel("Predicted FU3 Apathy Score")
    ax.set_title(f"Exp B — Baseline→FU3  R²={r2_score(y,y_pred):.2f}")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "expB_predicted_vs_actual.png", dpi=150)
    plt.close()

    # SHAP
    xgb_pipe.fit(X, y)
    shap_df = run_shap(xgb_pipe, X, feat_cols, label="ExpB")

    return {"metrics": metrics, "shap": shap_df}


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Loading data ===")
    df = load_and_clean(DATA_CSV)
    print(f"  Rows: {len(df)}  |  Patients: {df['patient_id'].nunique()}")
    print(f"  Apathy score range: {df['apathy_score'].min():.1f} – {df['apathy_score'].max():.1f}")

    results = {}
    results["experiment_a"] = experiment_a(df)
    results["experiment_b"] = experiment_b(df)

    # save metrics summary
    summary_rows = []
    for exp, res in results.items():
        if res is None:
            continue
        for model_key in ["xgb", "ridge", "metrics"]:
            if model_key in res and isinstance(res[model_key], dict):
                summary_rows.append({"experiment": exp, **res[model_key]})

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUT_DIR / "metrics_summary.csv", index=False)
    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(summary_df.to_string(index=False))
    print(f"\n✓ All outputs saved to {OUT_DIR}/")

    # reminder for small-N interpretation
    print("""
⚠  SMALL-SAMPLE NOTE (N≈16 patients):
   - R² and MAE from cross-validation are noisy estimates.
   - Treat SHAP rankings as hypothesis-generating, not confirmatory.
   - Consider supplementing with linear mixed models (see 05_mixed_models.py).
   - Do NOT report CV performance as generalisation performance.
""")
