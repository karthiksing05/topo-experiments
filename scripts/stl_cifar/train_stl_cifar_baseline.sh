#!/bin/bash
#SBATCH --job-name=stl_cifar_baseline
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH -t 24:00:00
#SBATCH -q inferno
#SBATCH --gres=gpu:A100:2
#SBATCH -o /storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/stl_cifar_baseline_%j.out
#SBATCH -e /storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/stl_cifar_baseline_%j.err

set -euo pipefail

EXPERIMENT_DIR="/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments"

# ---- CUPY cache (avoid home-dir quota) -------------------------------------
export CUPY_CACHE_DIR="${EXPERIMENT_DIR}/.cupy_cache"
mkdir -p "${CUPY_CACHE_DIR}"

# ---- Conda environment -----------------------------------------------------
source /storage/home/hcoda1/3/ksingara3/scratch/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ---- HuggingFace token (if needed) -----------------------------------------
if [ -f "${EXPERIMENT_DIR}/litcoder_token" ]; then
    export HUGGING_FACE_HUB_TOKEN="$(cat ${EXPERIMENT_DIR}/litcoder_token)"
    export HF_TOKEN="${HUGGING_FACE_HUB_TOKEN}"
fi
export HF_HOME="${EXPERIMENT_DIR}/huggingfacehub_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

# ---- Log directory ----------------------------------------------------------
mkdir -p "${EXPERIMENT_DIR}/scripts/logs"
mkdir -p "${EXPERIMENT_DIR}/outputs/stl_cifar/results"
mkdir -p "${EXPERIMENT_DIR}/outputs/stl_cifar/checkpoints"

# ---- Run --------------------------------------------------------------------
cd "${EXPERIMENT_DIR}"

echo "======================================================="
echo "  Job: stl_cifar_baseline"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "  GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "======================================================="

python src/stl_cifar/train_stl_cifar_baseline.py \
    --config configs/train_stl_cifar.json

echo "Job finished at $(date)"
