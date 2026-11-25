#!/bin/bash
#SBATCH --job-name=dpheno_eval_full
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_full_%j.out
#SBATCH --error=logs/eval_full_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irmak.ozarslan@unige.ch

# Run AFTER 01_meta_analysis.py has completed
# Evaluates the actual pipeline output in results/meta/

cd $SLURM_SUBMIT_DIR
mkdir -p logs

PYTHON=/home/users/o/ozarslan/dpheno-facial-action-unit/.venv/bin/python
set -a && source .env && set +a

echo "Python: $($PYTHON --version)"
echo "GEMINI_API_KEY: $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
echo "Node: $(hostname) | Start: $(date)"

# Check meta-analysis output exists
if [ ! -f "results/meta/raw_extractions.json" ]; then
    echo "ERROR: results/meta/raw_extractions.json not found"
    echo "Run sbatch scripts/run_meta.sbatch first"
    exit 1
fi

$PYTHON src/00_retrieval_eval.py --mode full

echo "Done: $(date)"
