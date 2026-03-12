#!/bin/bash
#SBATCH --job-name=stl_cifar_analysis
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH -t 00:30:00
#SBATCH -q inferno
#SBATCH -o /storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/stl_cifar_analysis_%j.out
#SBATCH -e /storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/stl_cifar_analysis_%j.err

set -euo pipefail

EXPERIMENT_DIR="/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments"

# ---- Conda environment -----------------------------------------------------
source /storage/home/hcoda1/3/ksingara3/scratch/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ---- Directory setup -------------------------------------------------------
mkdir -p "${EXPERIMENT_DIR}/scripts/logs"
mkdir -p "${EXPERIMENT_DIR}/outputs/stl_cifar/figures"

# ---- Run -------------------------------------------------------------------
cd "${EXPERIMENT_DIR}"

echo "======================================================="
echo "  Job: stl_cifar_analysis"
echo "  Date: $(date)"
echo "  Node: $(hostname)"
echo "======================================================="

# Pass --strict to fail fast if any training results are missing.
# Remove --strict to run analysis on whatever results are available.
python src/test/catastrophic_forgetting.py \
    --results-dir outputs/stl_cifar/results \
    --out-dir     outputs/stl_cifar/figures \
    --strict

echo ""
echo "Figures written to: ${EXPERIMENT_DIR}/outputs/stl_cifar/figures/"
ls -lh "${EXPERIMENT_DIR}/outputs/stl_cifar/figures/" 2>/dev/null || true
echo "Job finished at $(date)"
