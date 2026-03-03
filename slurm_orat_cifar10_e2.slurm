#!/bin/bash
#SBATCH --job-name=orat_c10_e2
#SBATCH --account=zghodsi
#SBATCH --partition=a10
#SBATCH --qos=normal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "========================================"
echo "ORAT CIFAR-10 epsilon=2/255"
echo "Job started on $(hostname)"
echo "Start time: $(date)"
echo "Working directory: $PWD"
echo "========================================"

# Use your conda environment Python directly
PYTHON="/home/pate2208/.conda/envs/2025.06-py313/orat/bin/python"

echo "Python version:"
$PYTHON --version

# Prevent CPU thread oversubscription
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Starting training..."

$PYTHON -u CIFAR10_ORAT.py \
  --epsilon 0.0078 \
  --dataset cifar10 \
  --lr 0.1 \
  --k 50000 \
  --m 1

echo "========================================"
echo "Training finished at $(date)"
echo "========================================"
