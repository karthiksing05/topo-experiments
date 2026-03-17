#!/bin/bash
#SBATCH --job-name=fmnist_forgetting_analysis
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=8GB
#SBATCH --cpus-per-task=2
#SBATCH -t 00:15:00
#SBATCH -q inferno
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/fmnist_forgetting_analysis-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/fmnist_forgetting_analysis-%j.err

set -euo pipefail

EXPERIMENT_DIR="/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments"

source /storage/home/hcoda1/3/ksingara3/scratch/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

mkdir -p "${EXPERIMENT_DIR}/scripts/logs"
mkdir -p "${EXPERIMENT_DIR}/outputs/fashion_mnist_forgetting/figures"

cd "${EXPERIMENT_DIR}"

echo "======================================================="
echo "  Job: fmnist_forgetting_analysis"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "======================================================="

python src/fashion_mnist_forgetting/analyze_fmnist_forgetting.py \
    --results  outputs/fashion_mnist_forgetting/results/fmnist_forgetting_results_latest.json \
    --out-dir  outputs/fashion_mnist_forgetting/figures \
    --strict

echo ""
echo "Figures written to: ${EXPERIMENT_DIR}/outputs/fashion_mnist_forgetting/figures/"
ls -lh "${EXPERIMENT_DIR}/outputs/fashion_mnist_forgetting/figures/" 2>/dev/null || true
echo "Job finished at $(date)"
