#!/bin/bash
#SBATCH --job-name=dpheno_meta
#SBATCH --partition=shared-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/meta_%j.out
#SBATCH --error=logs/meta_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irmak.ozarslan@unige.ch

cd $SLURM_SUBMIT_DIR
mkdir -p logs

PYTHON=/home/users/o/ozarslan/dpheno-facial-action-unit/.venv/bin/python

set -a && source .env && set +a

echo "Python:          $($PYTHON --version)"
echo "GEMINI_API_KEY:  $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
echo "Node: $(hostname) | GPU: $CUDA_VISIBLE_DEVICES | Start: $(date)"

$PYTHON src/01_meta_analysis.py

echo "Done: $(date)"
