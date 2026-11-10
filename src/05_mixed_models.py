"""
05_mixed_models.py
-------------------
Linear Mixed-Effects Models (LME) as a statistical complement to XGBoost.
With N≈16 patients, LME is often more reliable than XGBoost for inference.

Runs:
  1. Null model (random intercept per patient only) — baseline variance
  2. Univariate LMEs — each AU feature vs apathy, controlling for timepoint
  3. Multivariate LME — top features from SHAP, with random intercept per patient
  4. Trajectory model — does apathy change over time, and do AU features
     interact with timepoint?

Input:
  data/features.csv
  results/model/shap_ranking_ExpA.csv   (from 04_model_shap.py)

Output:
  results/lme/

pip install pandas numpy statsmodels matplotlib seaborn
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from pathlib import Path

DATA_CSV   = Path("data/features.csv")
SHAP_CSV   = Path("results/model/shap_ranking_ExpA.csv")
OUT_DIR    = Path("results/lme")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_N_FEATURES  = 8    # from SHAP ranking to use in multivariate model
MIN_VARIANCE    = 1e-6  # skip near-constant features


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["apathy_score"])
    # numeric timepoint
    tp_map = {"Baseline": 0, "FU1": 1, "FU3": 2}
    df["time_num"] = df["timepoint"].map(tp_map).fillna(0).astype(float)
    return df


def sanitize_col(name: str) -> str:
    """Make column name safe for statsmodels formula."""
    return (name.replace("-", "_")
                .replace(" ", "_")
                .replace(".", "_")
                .replace("(", "")
                .replace(")", ""))


def get_top_features(df: pd.DataFrame, n: int = TOP_N_FEATURES) -> list[str]:
    """Use SHAP ranking if available; fall back to univariate correlation."""
    feat_cols = [c for c in df.columns
                 if c not in ("patient_id", "timepoint", "apathy_score", "time_num")]

    if SHAP_CSV.exists():
        shap_df = pd.read_csv(SHAP_CSV)
        # filter to features that exist in current df
        candidates = shap_df[shap_df["feature"].isin(feat_cols)]["feature"].tolist()
        return candidates[:n]

    # fallback: Spearman r with apathy
    from scipy import stats
    sub = df.dropna(subset=["apathy_score"])
    corrs = {}
    for col in feat_cols:
        try:
            vals = sub[col].dropna()
            if vals.var() < MIN_VARIANCE or len(vals) < 5:
                continue
            r, _ = stats.spearmanr(sub.loc[vals.index, "apathy_score"], vals)
            corrs[col] = abs(r)
        except Exception:
            pass
    return sorted(corrs, key=lambda k: corrs[k], reverse=True)[:n]


# ── 1. Null model ──────────────────────────────────────────────────────────────
def null_model(df: pd.DataFrame) -> None:
    model = smf.mixedlm("apathy_score ~ 1", df,
                         groups=df["patient_id"])
    result = model.fit(reml=True)
    icc = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale)
    print(f"\n--- NULL MODEL ---")
    print(f"  Random-intercept variance (between-patient): {result.cov_re.iloc[0,0]:.3f}")
    print(f"  Residual variance:                           {result.scale:.3f}")
    print(f"  ICC (% variance explained by patient):       {icc:.1%}")
    print("  → High ICC means patient ID matters a lot (expected).")


# ── 2. Univariate LMEs ────────────────────────────────────────────────────────
def univariate_lmes(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each AU feature: apathy ~ feature + time_num + (1|patient_id)
    Returns sorted table of beta, SE, p-value.
    """
    feat_cols = [c for c in df.columns
                 if c not in ("patient_id", "timepoint", "apathy_score", "time_num")]

    rows = []
    for raw_col in feat_cols:
        col = sanitize_col(raw_col)
        df_tmp = df.rename(columns={raw_col: col}).copy()

        if df_tmp[col].var() < MIN_VARIANCE:
            continue
        if df_tmp[col].isna().mean() > 0.5:
            continue

        df_tmp = df_tmp.dropna(subset=[col])
        if len(df_tmp) < 10:
            continue

        try:
            formula = f"apathy_score ~ {col} + time_num"
            m = smf.mixedlm(formula, df_tmp, groups=df_tmp["patient_id"])
            r = m.fit(reml=False)   # REML=False needed for LRT comparisons
            rows.append({
                "feature":  raw_col,
                "beta":     r.params[col],
                "se":       r.bse[col],
                "z":        r.tvalues[col],
                "p_value":  r.pvalues[col],
                "n":        int(r.nobs),
            })
        except Exception:
            continue

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df

    # FDR correction (Benjamini-Hochberg) — important at small N
    from statsmodels.stats.multitest import multipletests
    _, p_fdr, _, _ = multipletests(result_df["p_value"], method="fdr_bh")
    result_df["p_fdr"] = p_fdr
    result_df = result_df.sort_values("p_value")

    # save
    result_df.to_csv(OUT_DIR / "univariate_lme.csv", index=False)
    print(f"\n--- UNIVARIATE LMEs ({len(result_df)} features) ---")
    sig = result_df[result_df["p_fdr"] < 0.05]
    print(f"  FDR-significant features: {len(sig)}")
    print(result_df.head(10)[["feature", "beta", "se", "p_value", "p_fdr"]].to_string(index=False))
    return result_df


# ── 3. Multivariate LME ───────────────────────────────────────────────────────
def multivariate_lme(df: pd.DataFrame, top_features: list[str]) -> None:
    print(f"\n--- MULTIVARIATE LME ---")
    print(f"  Features: {top_features}")

    # rename cols to be formula-safe
    rename_map = {f: sanitize_col(f) for f in top_features}
    df_tmp = df.rename(columns=rename_map).copy()
    safe_feats = list(rename_map.values())

    df_tmp = df_tmp.dropna(subset=safe_feats + ["apathy_score"])
    if len(df_tmp) < 10:
        print("  ⚠ Too few complete cases for multivariate model")
        return

    formula = "apathy_score ~ " + " + ".join(safe_feats) + " + time_num"
    try:
        model  = smf.mixedlm(formula, df_tmp, groups=df_tmp["patient_id"])
        result = model.fit(reml=True)
        print(result.summary())

        # coefficient plot
        params = result.params.drop(["Intercept", "Group Var"], errors="ignore")
        ci     = result.conf_int().drop(["Intercept", "Group Var"], errors="ignore")
        pvals  = result.pvalues.drop(["Intercept", "Group Var"], errors="ignore")

        fig, ax = plt.subplots(figsize=(8, max(4, len(params) * 0.4 + 1)))
        colors = ["#D65F5F" if p < 0.05 else "#888888" for p in pvals]
        y_pos  = range(len(params))
        ax.barh(list(y_pos), params.values, color=colors,
                xerr=[(params.values - ci.iloc[:, 0].values),
                      (ci.iloc[:, 1].values - params.values)],
                error_kw={"linewidth": 1.5, "capsize": 4})
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(params.index)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_xlabel("Fixed-effect coefficient")
        ax.set_title("Multivariate LME — fixed effects\n(red = p<0.05)")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "multivariate_lme_coefs.png", dpi=150)
        plt.close()

        with open(OUT_DIR / "multivariate_lme_summary.txt", "w") as f:
            f.write(str(result.summary()))

    except Exception as e:
        print(f"  ⚠ Multivariate model failed: {e}")


# ── 4. Trajectory model: does change in AU predict change in apathy? ──────────
def trajectory_model(df: pd.DataFrame, feature: str) -> None:
    """
    Interaction model: apathy ~ feature * time_num + (1|patient_id)
    Tests whether the AU feature changes with time alongside apathy.
    """
    col = sanitize_col(feature)
    df_tmp = df.rename(columns={feature: col}).dropna(subset=[col, "apathy_score"])
    if len(df_tmp) < 10:
        return

    try:
        formula = f"apathy_score ~ {col} * time_num"
        model   = smf.mixedlm(formula, df_tmp, groups=df_tmp["patient_id"])
        result  = model.fit(reml=True)
        inter_p = result.pvalues.get(f"{col}:time_num", np.nan)
        print(f"  Trajectory interaction [{feature} × time]:  p={inter_p:.3f}")
    except Exception as e:
        print(f"  ⚠ Trajectory model failed for {feature}: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Loading data ===")
    df = load(DATA_CSV)
    print(f"  Rows: {len(df)}  |  Patients: {df['patient_id'].nunique()}")

    null_model(df)
    uni_df = univariate_lmes(df)

    top_features = get_top_features(df)
    if not top_features:
        print("  ⚠ No features found — check SHAP CSV or feature columns")
    else:
        multivariate_lme(df, top_features)

        print("\n--- TRAJECTORY INTERACTIONS (top 3 features) ---")
        for f in top_features[:3]:
            trajectory_model(df, f)

    print(f"\n✓ LME outputs saved to {OUT_DIR}/")
    print("""
INTERPRETATION GUIDE:
  - ICC > 0.5 → patient identity dominates → GroupKFold was the right CV choice
  - Univariate p-values: treat FDR < 0.05 as a shortlist, not proof
  - Multivariate coefficients: direction more trustworthy than magnitude at N≈16
  - Trajectory interaction (feature × time): tests whether AU tracks apathy change
""")
