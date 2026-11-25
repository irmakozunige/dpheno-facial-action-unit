#!/bin/bash
#SBATCH --job-name=dpheno_eval
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irmak.ozarslan@unige.ch

cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Use venv Python directly — bypasses conda activate issues on compute nodes
PYTHON=/home/users/o/ozarslan/dpheno-facial-action-unit/.venv/bin/python

# Load API keys
set -a && source .env && set +a

echo "Python:          $($PYTHON --version)"
echo "GEMINI_API_KEY:  $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | Start: $(date)"

$PYTHON src/00_retrieval_eval.py

echo "Done: $(date)"
