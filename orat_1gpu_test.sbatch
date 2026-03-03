#!/bin/bash
#SBATCH -A zghodsi
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -J orat_1gpu

# Put logs on SCRATCH (not HOME)
#SBATCH -o /scratch/gilbreth/%u/orat/logs/%x_%j.out
#SBATCH -e /scratch/gilbreth/%u/orat/logs/%x_%j.err

set -euo pipefail
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
date

# --- scratch layout ---
export ORAT_SCRATCH=/scratch/gilbreth/$USER/orat
export OUT_DIR=$ORAT_SCRATCH/runs/cifar10_resnet18_1gpu
export DATA_DIR=$ORAT_SCRATCH/data
export CACHE_DIR=$ORAT_SCRATCH/cache
mkdir -p "$ORAT_SCRATCH"/{logs,runs,data,cache,ckpts} "$OUT_DIR"

# --- keep caches off HOME ---
export TMPDIR=$CACHE_DIR/tmp
export XDG_CACHE_HOME=$CACHE_DIR/xdg
export TORCH_HOME=$CACHE_DIR/torch
export HF_HOME=$CACHE_DIR/huggingface
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$HF_HOME"

# --- activate your conda env (adjust if needed) ---
source ~/.bashrc
conda activate /home/pate2208/.conda/envs/2025.06-py313/orat

# Quick sanity checks
nvidia-smi -L
python - <<'PY'
import torch, torchvision
print("python ok")
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device0:", torch.cuda.get_device_name(0))
PY

# --- run ORAT training ---
cd /home/pate2208/orat

python CIFAR10_ORAT.py \
  --dataset cifar10 \
  --net resnet18 \
  --gpuid 0 \
  --out_dir "$OUT_DIR" \
  --data_dir "$DATA_DIR"

echo "Done: $(date)"
