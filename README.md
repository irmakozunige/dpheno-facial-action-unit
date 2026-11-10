# dpheno-facial-action-unit

Automated pipeline for identifying facial Action Unit (AU) biomarkers of apathy in schizophrenia using clinical interview videos.

## Overview

```
Clinical Videos → OpenFace AU Extraction → XGBoost + SHAP → Apathy Biomarkers
       ↑
LLM Meta-Analysis (PubMed, Semantic Scholar, Google Scholar, CrossRef)
informs which AUs to extract
```

## Pipeline

| Script | Purpose |
|--------|---------|
| `src/01_meta_analysis.py` | Multi-source literature retrieval + forest plots + LLM narrative |
| `src/02_feature_extraction.py` | OpenFace wrapper → per-video AU statistics |
| `src/03_eda.py` | Apathy trajectories, AU heatmaps, correlation plots |
| `src/04_model_shap.py` | XGBoost + SHAP (GroupKFold by patient) |
| `src/05_mixed_models.py` | Linear mixed-effects models (statistical complement) |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### OpenFace (required for step 2)
```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make -j4
# set OPENFACE_BIN in 02_feature_extraction.py
```

## Data Structure

```
data/
├── videos/
│   ├── P01_Baseline.mp4
│   ├── P01_FU1.mp4
│   └── P01_FU3.mp4
└── labels.csv          # patient_id, timepoint, apathy_score
```

### labels.csv format
```
patient_id,timepoint,apathy_score
P01,Baseline,32
P01,FU3,24
P02,Baseline,28
```
Timepoints: `Baseline`, `FU1`, `FU3`

## Run Order

```bash
python src/01_meta_analysis.py    # → review results/meta/ before continuing
python src/02_feature_extraction.py
python src/03_eda.py
python src/04_model_shap.py
python src/05_mixed_models.py
```

## Literature Sources (Step 1)

- **PubMed** via Biopython Entrez
- **Semantic Scholar** via official API
- **Google Scholar** via `scholarly`
- **CrossRef** (covers Nature, Wiley, Frontiers, Elsevier/ScienceDirect DOI metadata)

> Full-text access to ScienceDirect/Wiley requires institutional credentials.
> The pipeline extracts from abstracts; manually add means/SDs to `results/meta/au_long.csv` after reading full texts.

## Small-N Notes (N≈16)

- GroupKFold CV prevents patient leakage
- XGBoost capped at `max_depth=2` with strong L1/L2
- SHAP rankings are hypothesis-generating, not confirmatory
- Supplement with LME results (script 05) for reviewer robustness

## Outputs

```
results/
├── meta/
│   ├── raw_extractions.json
│   ├── au_long.csv
│   ├── meta_results.csv
│   ├── forest_AU**.png       ← one per AU
│   ├── bubble_summary.png
│   ├── direction_heatmap.png
│   └── meta_analysis_summary.txt
├── eda/
└── model/
    ├── shap_beeswarm_ExpA.png
    ├── shap_bar_ExpA.png
    └── metrics_summary.csv
```
