#!/bin/bash
#SBATCH --job-name=dpheno_meta
#SBATCH --partition=cpu          # ← change to your partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/meta_%j.out
#SBATCH --error=logs/meta_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irmak.ozarslan@unige.ch

cd $SLURM_SUBMIT_DIR
mkdir -p logs

source ~/.bashrc
conda activate dpheno-au

set -a && source .env && set +a
echo "GEMINI_API_KEY set: $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
echo "Node: $(hostname) | Start: $(date)"

python src/01_meta_analysis.py

echo "Done: $(date)"
