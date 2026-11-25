#!/bin/bash
#SBATCH --job-name=dpheno_eval
#SBATCH --partition=cpu          # ← change to your partition name
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irmak.ozarslan@unige.ch

cd $SLURM_SUBMIT_DIR
mkdir -p logs

source ~/.bashrc
conda activate dpheno-au

# load API keys
set -a && source .env && set +a
echo "GEMINI_API_KEY set: $([ -n "$GEMINI_API_KEY" ] && echo YES || echo NO)"
echo "Node: $(hostname) | Start: $(date)"

python src/00_retrieval_eval.py

echo "Done: $(date)"
