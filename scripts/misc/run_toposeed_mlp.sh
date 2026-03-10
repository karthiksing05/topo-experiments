#!/bin/bash
#SBATCH --job-name=run_toposeed_mlp
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 6:00:00
#SBATCH -qinferno
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/run_toposeed_mlp-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/run_toposeed_mlp-%j.err

# ---- Environment ------------------------------------------------------------
module purge
source /storage/home/hcoda1/3/ksingara3/scratch/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate topovlm

EXPERIMENT_DIR="/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments"

# CONFIG can be overridden via: sbatch --export=CONFIG=configs/mlp_fashion_mnist/mlp_fashion_mnist_aggressive_growth.json run_toposeed_mlp.sh
CONFIG="${CONFIG:-configs/mlp_fashion_mnist/mlp_fashion_mnist.json}"

export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

# ---- Run --------------------------------------------------------------------
cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs

# Derive a short name from the config filename for logging clarity
CONFIG_NAME="$(basename "$CONFIG" .json)"

echo "=== TopoSeed MLP on Fashion-MNIST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
echo "Config: $CONFIG  ($CONFIG_NAME)"
date

srun python -u src/misc/mlp_fashion_mnist.py \
    --config "$CONFIG" \
    --device cuda

echo "Done: $CONFIG_NAME"
date
