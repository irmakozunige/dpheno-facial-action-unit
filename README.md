# dpheno-facial-action-unit

Automated pipeline for identifying facial Action Unit (AU) biomarkers of apathy in schizophrenia using clinical interview videos.

## Overview

```
Clinical Videos → OpenFace AU Extraction → XGBoost + SHAP → Apathy Biomarkers
       ↑
LLM Meta-Analysis
  ├── PubMed · Semantic Scholar · Google Scholar · CrossRef
  ├── Full-text via Unpaywall (OA) + EZproxy (institutional)
  ├── Semantic AU labelling (anatomical label, face region, functional tag)
  └── Forest plots + feature vocabulary table → YOU decide which AUs to extract
```

## Pipeline

| Script | Purpose |
|--------|---------|
| `src/01_meta_analysis.py` | Multi-source retrieval + full-text + semantic AU vocab + forest plots |
| `src/02_feature_extraction.py` | OpenFace wrapper → per-video AU statistics |
| `src/03_eda.py` | Apathy trajectories, AU heatmaps, correlation plots |
| `src/04_model_shap.py` | XGBoost + SHAP (GroupKFold by patient) |
| `src/05_mixed_models.py` | Linear mixed-effects models |

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

### Institutional full-text access (EZproxy)
```bash
export EZPROXY_BASE="https://ezproxy.youruniversity.edu/login?url="
export EZPROXY_COOKIE="your_session_cookie"
# Get cookie: browser DevTools → Application → Cookies after EZproxy login
```
Without these, the pipeline uses Unpaywall (open-access only) and falls back to abstracts.

### OpenFace (required for step 2)
```bash
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make -j4
```

## Literature Sources (Step 1)

| Source | Coverage |
|--------|---------|
| PubMed | Core clinical/neuroscience literature |
| Semantic Scholar | Nature, Frontiers, Springer, broad CS+neuro |
| Google Scholar | Widest coverage — ScienceDirect, Wiley, everything |
| CrossRef | DOI metadata — Wiley, Elsevier, Frontiers, Sage |
| Unpaywall | Open-access full-text PDFs (auto) |
| EZproxy | Institutional full-text PDFs (requires credentials) |

## AU Semantic Mapping

For each AU found in the literature, Claude extracts:

| Field | Example |
|-------|---------|
| `au` | AU12 |
| `au_label` | Lip Corner Puller |
| `face_region` | mouth |
| `functional_tag` | smile / positive affect |
| `direction` | decreased (in SZ vs HC) |
| `feature_type` | intensity / frequency / duration / presence |
| `hedges_g` | −0.84 |

After running Step 1, open **`results/meta/au_feature_vocab.csv`** and fill in the `recommended_feature` and `include_in_model` columns. This becomes your feature specification for Step 2.

## Data Structure

```
data/
├── videos/
│   ├── P01_Baseline.mp4
│   ├── P01_FU1.mp4
│   └── P01_FU3.mp4
└── labels.csv      # patient_id, timepoint, apathy_score
```

## Run Order

```bash
python src/01_meta_analysis.py    # → review results/meta/ — especially au_feature_vocab.csv
# edit AU_INTENSITIES in 02_feature_extraction.py based on your review
python src/02_feature_extraction.py
python src/03_eda.py
python src/04_model_shap.py
python src/05_mixed_models.py
```

## Outputs (Step 1)

```
results/meta/
├── raw_extractions.json       per-paper Claude extraction
├── au_long.csv                one row per paper × AU with effect sizes
├── au_feature_vocab.csv  ★   MASTER TABLE — review and annotate this
├── meta_results.csv           pooled Hedges' g per AU
├── forest_AU**.png            one forest plot per AU (≥2 studies)
├── bubble_summary.png         effect size × n studies × heterogeneity
├── direction_heatmap.png      study × AU direction matrix
├── vocab_heatmap.png          AU × face region evidence chart
└── meta_analysis_summary.txt  LLM narrative
```

## Small-N Notes (N≈16 patients)

- GroupKFold CV prevents patient leakage across train/test
- XGBoost: `max_depth=2`, strong L1/L2 regularisation
- SHAP rankings are hypothesis-generating, not confirmatory
- Supplement with LME results (script 05) for reviewer robustness
