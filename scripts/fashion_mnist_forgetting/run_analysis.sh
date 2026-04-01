#!/bin/bash
#SBATCH --job-name=fmnist_forgetting_analysis
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --time=01:00:00
#SBATCH --qos=short
#SBATCH --output=slurm/slurm_outputs/fmnist_forgetting_analysis-%j.out
#SBATCH --error=slurm/slurm_errors/fmnist_forgetting_analysis-%j.err

set -euo pipefail

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

mkdir -p slurm/slurm_outputs slurm/slurm_errors
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
