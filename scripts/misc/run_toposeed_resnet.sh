#!/bin/bash
#SBATCH --job-name=run_toposeed_resnet
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 12:00:00
#SBATCH -qinferno
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/run_toposeed_resnet-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/run_toposeed_resnet-%j.err

# ---- Environment ------------------------------------------------------------
module purge
source /storage/home/hcoda1/3/ksingara3/scratch/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate topovlm

EXPERIMENT_DIR="/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments"

export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

# ---- Run --------------------------------------------------------------------
cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs

echo "=== TopoSeed ResNet-18 on CIFAR-10 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
echo "GPUs:   $CUDA_VISIBLE_DEVICES"
date

srun python -u src/misc/resnet_cifar10.py \
    --config configs/resnet_cifar10.json \
    --device cuda

echo "Done."
date
