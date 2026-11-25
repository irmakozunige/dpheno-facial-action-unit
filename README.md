# dpheno-facial-action-unit

Automated pipeline for identifying facial Action Unit (AU) biomarkers of apathy in schizophrenia using clinical interview videos.

## Overview

```
Clinical Videos → OpenFace AU Extraction → XGBoost + SHAP → Apathy Biomarkers
       ↑
LLM Meta-Analysis (8 free sources, full-text, provenance-anchored)
       ↑
Retrieval Evaluation (RAGAS-style, Gemini judge)
```

## Pipeline

| Script | Purpose | Run via |
|--------|---------|---------|
| `src/00_retrieval_eval.py` | Evaluate retrieval quality before full run | SLURM |
| `src/01_meta_analysis.py` | Multi-source retrieval + forest plots + AU vocab | SLURM |
| `src/02_feature_extraction.py` | OpenFace wrapper → per-video AU statistics | local |
| `src/03_eda.py` | Apathy trajectories, AU heatmaps, correlations | local |
| `src/04_model_shap.py` | XGBoost + SHAP (GroupKFold by patient) | local |
| `src/05_mixed_models.py` | Linear mixed-effects models | local |

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/irmakozunige/dpheno-facial-action-unit.git
cd dpheno-facial-action-unit
conda create -n dpheno-au python=3.11
conda activate dpheno-au
pip install -r requirements.txt
```

### 2. API keys

```bash
cp .env.example .env
```

Open `.env` and fill in your keys:

```
GEMINI_API_KEY=AIzaSy...
```

**Gemini API key (free):**
1. Go to **aistudio.google.com**
2. Sign in with any Google account
3. Click "Get API key" → "Create API key"
4. Copy and paste into `.env`

Free limits: **1500 req/day, 15 req/min** (Flash) · **50 req/day** (Pro, used only for final narrative)

**Optional — CORE API key (more full-text coverage):**
1. Go to **core.ac.uk/api-keys**
2. Register and generate a key
3. Add `CORE_API_KEY=...` to `.env`

### 3. Load environment variables

```bash
set -a && source .env && set +a

# verify
echo $GEMINI_API_KEY    # should print your key
```

> Add `set -a && source .env && set +a` to your `~/.bashrc` to load automatically.

---

## Running on a cluster (SLURM)

Scripts 00 and 01 make outbound HTTP requests and must run on compute nodes, not the login node.

```bash
# check available partitions
sinfo -s

# edit partition name in both scripts
nano scripts/run_eval.sbatch    # change --partition=cpu
nano scripts/run_meta.sbatch    # change --partition=cpu

# submit evaluation first (30 min, ~60 Gemini calls)
sbatch scripts/run_eval.sbatch

# monitor
squeue -u $USER
tail -f logs/eval_<jobid>.out

# once eval looks good, submit full meta-analysis (4 hrs)
sbatch scripts/run_meta.sbatch
```

---

## Literature Sources (Step 1)

| Source | Coverage | Full text? |
|--------|---------|-----------|
| PubMed | Core clinical literature | Abstracts |
| Europe PMC | PMC-deposited papers | ✅ Full XML |
| Semantic Scholar | 200M+ papers | ✅ OA PDFs |
| OpenAlex | Largest open graph | ✅ OA PDFs |
| CORE.ac.uk | 200M+ OA full texts | ✅ Full PDFs |
| CrossRef | Wiley, Elsevier, Frontiers | Abstracts |
| bioRxiv / medRxiv | Preprints | ✅ Full PDFs |
| Google Scholar | Widest coverage | Abstracts + some PDFs |
| Unpaywall | OA PDF finder | ✅ OA PDFs |
| Publisher direct | citation_pdf_url meta tag | ✅ Where exposed |
| Sci-Hub | Last resort | ✅ Full PDFs |

---

## Retrieval Evaluation (Step 0)

Runs on a 20-paper sample before the full meta-analysis. Uses Gemini 2.0 Flash as judge.

| Metric | Target | What it catches |
|--------|--------|----------------|
| Context Relevance | > 0.6 | Off-topic chunks |
| Faithfulness | > 0.7 | Hallucinated AU values |
| Answer Relevance | > 0.6 | Off-target extractions |
| Full-text hit rate | > 0.4 | PDF retrieval coverage |
| AU keyword hit rate | > 0.5 | Chunks missing AU codes |
| Chunk dedup rate | < 0.3 | Redundancy across sources |

Outputs in `results/eval/`:
- `retrieval_eval.csv` — per-paper scores
- `retrieval_eval.png` — radar chart + source comparison
- `eval_report.txt` — LLM narrative with recommendations

**After manually reading 5–10 papers**, fill in `GOLD_SET` at the top of `00_retrieval_eval.py` to unlock Context Recall and Precision@AU metrics.

---

## AU Semantic Mapping

For each AU found in the literature, the pipeline extracts:

| Field | Example |
|-------|---------|
| `au` | AU12 |
| `au_label` | Lip Corner Puller |
| `face_region` | mouth |
| `functional_tag` | smile / positive affect |
| `direction` | decreased (in SZ vs HC) |
| `feature_type` | intensity / frequency / duration / presence |
| `hedges_g` | −0.84 |
| `exact_quote` | "AU12 intensity was significantly lower..." |
| `source_page` | 4 |
| `source_para` | 2 |
| `doi_link` | https://doi.org/10.1016/... |
| `pubmed_link` | https://pubmed.ncbi.nlm.nih.gov/... |

After running Step 1, open **`results/meta/au_feature_vocab.csv`**, fill in `recommended_feature` and `include_in_model` columns, then update `AU_INTENSITIES` in `02_feature_extraction.py`.

---

## Data Structure

```
data/
├── videos/
│   ├── P01_Baseline.mp4
│   ├── P01_FU1.mp4        ← only if patient has FU1
│   └── P01_FU3.mp4        ← only if patient has FU3
└── labels.csv
```

### labels.csv format

```
patient_id,timepoint,apathy_score
P01,Baseline,32
P01,FU3,24
P02,Baseline,28
P02,FU1,25
P02,FU3,19
```

Timepoints: `Baseline`, `FU1`, `FU3`

---

## Full Run Order

```bash
# 0. Evaluate retrieval on 20 papers (SLURM)
sbatch scripts/run_eval.sbatch
# → review results/eval/ — fix any issues before continuing

# 1. Full meta-analysis (SLURM)
sbatch scripts/run_meta.sbatch
# → review results/meta/au_feature_vocab.csv
# → fill recommended_feature + include_in_model columns
# → update AU_INTENSITIES in src/02_feature_extraction.py

# 2–5. Run locally after reviewing the literature output
python src/02_feature_extraction.py   # requires OpenFace on PATH
python src/03_eda.py
python src/04_model_shap.py
python src/05_mixed_models.py
```

---

## OpenFace Installation

```bash
sudo apt-get install libopenblas-dev liblapack-dev
git clone https://github.com/TadasBaltrusaitis/OpenFace.git
cd OpenFace && mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE .. && make -j4
# set OPENFACE_BIN in src/02_feature_extraction.py to full path
```

---

## Small-N Notes (N ≈ 16 patients)

| Risk | Mitigation |
|------|-----------|
| Patient leakage across train/test | GroupKFold CV by patient_id |
| XGBoost overfitting | max_depth=2, strong L1/L2 |
| Inflated p-values | FDR correction in LMEs |
| SHAP instability | Treat as hypothesis-generating only |
| Experiment B underpowered | Min completer check; skips gracefully |

---

## LLM Backend Options

| Backend | Cost | Setup |
|---------|------|-------|
| **Gemini 2.0 Flash** (default) | Free (1500/day) | `GEMINI_API_KEY` in `.env` |
| Ollama (local) | Free | `LLM_BACKEND=openai_compat` + `LLM_BASE_URL=http://localhost:11434/v1` |
| vLLM on cluster | Free (GPU) | `LLM_BACKEND=openai_compat` + `LLM_BASE_URL=http://localhost:8000/v1` |
| Anthropic Claude | Paid | `LLM_BACKEND=anthropic` + `LLM_API_KEY=sk-ant-...` |

---

## Outputs

```
results/
├── eval/
│   ├── retrieval_eval.csv
│   ├── retrieval_eval.png
│   └── eval_report.txt
├── meta/
│   ├── raw_extractions.json
│   ├── au_long.csv              ← one row per paper × AU, fully traceable
│   ├── au_feature_vocab.csv  ★ ← REVIEW AND ANNOTATE THIS
│   ├── meta_results.csv
│   ├── forest_AU**.png
│   ├── bubble_summary.png
│   ├── direction_heatmap.png
│   ├── vocab_heatmap.png
│   └── meta_analysis_summary.txt
├── eda/
└── model/
    ├── shap_beeswarm_ExpA.png
    ├── shap_bar_ExpA.png
    └── metrics_summary.csv
```
