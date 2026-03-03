#!/bin/bash
#SBATCH --job-name=orat_c10_e8
#SBATCH --account=zghodsi
#SBATCH --partition=a30
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Running in $PWD"

# Ensure we use your conda env's python (no need to `conda activate` inside batch)
PYTHON="/home/pate2208/.conda/envs/2025.06-py313/orat/bin/python"
$PYTHON --version

# Optional: reduce CPU thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Run ORAT CIFAR-10 (epsilon=8/255)
$PYTHON -u CIFAR10_ORAT.py \
  --epsilon 0.031 \
  --step_size 0.00775 \
  --dataset cifar10 \
  --lr 0.1 \
  --k 50000 \
  --m 1

echo "Job finished at $(date)"
