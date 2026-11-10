"""
02_feature_extraction.py
-------------------------
Runs OpenFace on all videos, then aggregates per-video AU statistics
into a single flat feature CSV ready for modelling.

Folder structure assumed:
    data/
        videos/
            P01_Baseline.mp4
            P01_FU1.mp4
            P01_FU3.mp4
            P02_Baseline.mp4
            ...
        labels.csv          ← patient_id, timepoint, apathy_score (PANSS / BNSS)

Output:
    data/features.csv       ← one row per (patient, timepoint)

Requirements:
    - OpenFace installed; FeatureExtraction binary on PATH
      (https://github.com/TadasBaltrusaitis/OpenFace)
    pip install pandas numpy tqdm
"""

import os, re, subprocess
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
VIDEO_DIR      = Path("data/videos")
OPENFACE_OUT   = Path("data/openface_raw")
FEATURES_OUT   = Path("data/features.csv")
LABELS_CSV     = Path("data/labels.csv")
OPENFACE_BIN   = "FeatureExtraction"   # or full path e.g. ~/OpenFace/build/bin/FeatureExtraction

# AUs to include — update this list AFTER your meta-analysis
# These are the most commonly reported in the schizophrenia literature
AU_INTENSITIES = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
                  "AU09", "AU10", "AU12", "AU14", "AU15", "AU17",
                  "AU20", "AU23", "AU25", "AU26", "AU45"]

AU_OCCURRENCES = [f"{au}_c" for au in AU_INTENSITIES]  # binary presence

GAZE_COLS      = ["gaze_angle_x", "gaze_angle_y"]
POSE_COLS      = ["pose_Rx", "pose_Ry", "pose_Rz"]

OPENFACE_OUT.mkdir(parents=True, exist_ok=True)

# ── 1. Run OpenFace ───────────────────────────────────────────────────────────
def run_openface(video_path: Path) -> Path:
    """Run OpenFace FeatureExtraction on one video; return path to output CSV."""
    out_dir = OPENFACE_OUT / video_path.stem
    out_dir.mkdir(exist_ok=True)
    csv_out = out_dir / f"{video_path.stem}.csv"

    if csv_out.exists():
        return csv_out   # already processed

    cmd = [
        OPENFACE_BIN,
        "-f", str(video_path),
        "-out_dir", str(out_dir),
        "-aus",          # action units
        "-gaze",         # gaze direction
        "-pose",         # head pose
        "-2Dfp",         # 2D facial landmarks (optional, used for QC)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ OpenFace failed on {video_path.name}:\n{result.stderr[:300]}")
    return csv_out


# ── 2. Aggregate frame-level → video-level features ──────────────────────────
def aggregate_openface_csv(csv_path: Path) -> dict:
    """
    Returns a flat dict of statistical summaries over valid (confident) frames.
    For each AU intensity: mean, std, max, % frames active (>0.5 intensity).
    For each AU occurrence: proportion of frames where AU was present.
    For gaze/pose: mean + std.
    Also adds higher-level features:
      - n_valid_frames, mean_confidence
      - expressivity_index: mean of all AU intensities (overall facial movement)
      - temporal_variability: mean frame-to-frame delta of AU intensities
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # keep only confident frames (OpenFace confidence > 0.8)
    if "confidence" in df.columns:
        df_valid = df[df["confidence"] > 0.8].copy()
    else:
        df_valid = df.copy()

    if len(df_valid) < 30:   # less than 1 second at 30fps — skip
        print(f"  ⚠ Too few confident frames in {csv_path.name} ({len(df_valid)})")
        return {}

    feats = {
        "n_valid_frames":  len(df_valid),
        "mean_confidence": df_valid["confidence"].mean() if "confidence" in df_valid else np.nan,
    }

    # AU intensities (columns named AU01_r, AU04_r …)
    au_intensity_cols = [c for c in df_valid.columns
                         if re.match(r"AU\d{2}_r", c)]
    if au_intensity_cols:
        for col in au_intensity_cols:
            au = col.replace("_r", "")
            vals = df_valid[col].values
            feats[f"{au}_mean"]   = vals.mean()
            feats[f"{au}_std"]    = vals.std()
            feats[f"{au}_max"]    = vals.max()
            feats[f"{au}_pct_active"] = (vals > 0.5).mean()

        # expressivity: mean intensity across all AUs
        feats["expressivity_index"] = df_valid[au_intensity_cols].mean(axis=1).mean()

        # temporal variability: mean abs frame-to-frame change
        deltas = df_valid[au_intensity_cols].diff().abs().mean(axis=1)
        feats["temporal_variability"] = deltas.mean()

    # AU occurrences (columns named AU01_c …)
    au_occ_cols = [c for c in df_valid.columns if re.match(r"AU\d{2}_c", c)]
    for col in au_occ_cols:
        feats[f"{col.replace('_c','')}_occ_rate"] = df_valid[col].mean()

    # Gaze
    for col in GAZE_COLS:
        if col in df_valid.columns:
            feats[f"{col}_mean"] = df_valid[col].mean()
            feats[f"{col}_std"]  = df_valid[col].std()

    # Head pose
    for col in POSE_COLS:
        if col in df_valid.columns:
            feats[f"{col}_mean"] = df_valid[col].mean()
            feats[f"{col}_std"]  = df_valid[col].std()

    return feats


# ── 3. Parse filename → patient_id + timepoint ───────────────────────────────
def parse_filename(stem: str) -> tuple[str, str]:
    """
    Expected naming: P01_Baseline, P01_FU1, P01_FU3
    Returns (patient_id, timepoint)
    """
    parts = stem.split("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, "Unknown"


# ── 4. Build feature dataframe ────────────────────────────────────────────────
def build_feature_df() -> pd.DataFrame:
    videos = sorted(VIDEO_DIR.glob("*.mp4")) + \
             sorted(VIDEO_DIR.glob("*.avi")) + \
             sorted(VIDEO_DIR.glob("*.mov"))

    if not videos:
        raise FileNotFoundError(f"No videos found in {VIDEO_DIR}")

    rows = []
    for vp in tqdm(videos, desc="Processing videos"):
        pid, tp = parse_filename(vp.stem)
        print(f"\n  Processing {vp.name} → patient={pid}, timepoint={tp}")

        csv_path = run_openface(vp)
        feats    = aggregate_openface_csv(csv_path)

        if feats:
            rows.append({"patient_id": pid, "timepoint": tp, **feats})

    df = pd.DataFrame(rows)
    return df


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Building feature matrix ===")
    feat_df = build_feature_df()

    # Merge with labels (apathy scores)
    if LABELS_CSV.exists():
        labels = pd.read_csv(LABELS_CSV)   # columns: patient_id, timepoint, apathy_score
        feat_df = feat_df.merge(labels, on=["patient_id", "timepoint"], how="left")
        missing = feat_df["apathy_score"].isna().sum()
        if missing:
            print(f"  ⚠ {missing} rows missing apathy score after merge — check labels.csv")
    else:
        print(f"  ⚠ {LABELS_CSV} not found — features saved without labels")

    feat_df.to_csv(FEATURES_OUT, index=False)
    print(f"\n✓ Features saved to {FEATURES_OUT}")
    print(f"  Shape: {feat_df.shape}")
    print(f"  Patients: {feat_df['patient_id'].nunique()}")
    print(f"  Timepoints: {feat_df['timepoint'].value_counts().to_dict()}")
    print(f"\nFeature columns ({feat_df.shape[1]-3}):")
    non_meta = [c for c in feat_df.columns
                if c not in ("patient_id", "timepoint", "apathy_score")]
    print("  " + ", ".join(non_meta[:20]) + ("..." if len(non_meta) > 20 else ""))
