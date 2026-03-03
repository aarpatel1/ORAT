#!/bin/bash
#SBATCH -A zghodsi
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH -J orat_c10_r20_noAA
#SBATCH -o /scratch/gilbreth/%u/orat/logs/%x_%j.out
#SBATCH -e /scratch/gilbreth/%u/orat/logs/%x_%j.err

set -eo pipefail

module load anaconda cuda
conda activate orat

# Keep data/runs/logs in scratch
BASE=/scratch/gilbreth/$USER/orat
mkdir -p "$BASE/logs" "$BASE/runs" "$BASE/data"

echo "==== Job info ===="
echo "JobID: ${SLURM_JOB_ID:-unset}"
echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
echo "==============="

# Optional: keep torchvision downloads in scratch (recommended)
export TORCH_HOME="$BASE/data"
export ORAT_DATA="$BASE/data"

# Run ORAT CIFAR10 + ResNet20
python -u CIFAR10_ORAT.py \
  --dataset cifar10 \
  --net resnet20 \
  --gpuid 0 \
  --epochs 120 \
  --out_dir "$BASE/runs/c10_r20_e120_noAA"
