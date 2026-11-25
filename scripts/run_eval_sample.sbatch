#!/bin/bash
#SBATCH --job-name=dpheno_eval_sample
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_sample_%j.out
#SBATCH --error=logs/eval_sample_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irmak.ozarslan@unige.ch

cd $SLURM_SUBMIT_DIR
mkdir -p logs

PYTHON=/home/users/o/ozarslan/dpheno-facial-action-unit/.venv/bin/python
set -a && source .env && set +a

echo "Python: $($PYTHON --version)"
echo "GEMINI_API_KEY: $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
echo "Node: $(hostname) | Start: $(date)"

$PYTHON src/00_retrieval_eval.py --mode sample

echo "Done: $(date)"
