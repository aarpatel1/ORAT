#!/bin/bash
#SBATCH -A zghodsi
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 08:00:00
#SBATCH -c 4
#SBATCH -J build_orat_img
#SBATCH -o /scratch/gilbreth/%u/orat/logs/%x_%j.out
#SBATCH -e /scratch/gilbreth/%u/orat/logs/%x_%j.err

set -euo pipefail

cd /home/$USER/orat

mkdir -p /scratch/gilbreth/$USER/orat/logs
mkdir -p /scratch/gilbreth/$USER/orat
mkdir -p /scratch/gilbreth/$USER/apptainer_cache
mkdir -p /scratch/gilbreth/$USER/apptainer_tmp

export APPTAINER_CACHEDIR=/scratch/gilbreth/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/gilbreth/$USER/apptainer_tmp

echo "Hostname: $(hostname)"
echo "Start time: $(date)"
apptainer --version

rm -f /scratch/gilbreth/$USER/orat/orat.sif
rm -f /scratch/gilbreth/$USER/orat/build_orat.log

apptainer build /scratch/gilbreth/$USER/orat/orat.sif orat.def 2>&1 | tee /scratch/gilbreth/$USER/orat/build_orat.log

echo "Build finished at: $(date)"
ls -lh /scratch/gilbreth/$USER/orat/orat.sif
