#!/bin/bash
#SBATCH -A zghodsi
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 24:00:00
#SBATCH -c 8
#SBATCH -J orat_c10_smallcnn_eps2
#SBATCH -o /scratch/gilbreth/%u/orat/logs/%x_%j.out
#SBATCH -e /scratch/gilbreth/%u/orat/logs/%x_%j.err

set -eo pipefail

module load anaconda cuda
conda activate orat

mkdir -p /scratch/gilbreth/$USER/orat/logs
mkdir -p /scratch/gilbreth/$USER/orat/runs

echo "Hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python -u CIFAR10_ORAT.py \
  --dataset cifar10 \
  --net smallcnn \
  --gpuid 0 \
  --epochs 120 \
  --epsilon 0.00784313725490196 \
  --num_steps 10 \
  --step_size 0.00196078431372549 \
  --out_dir /scratch/gilbreth/$USER/orat/runs/c10_smallcnn_eps2_255_e120
