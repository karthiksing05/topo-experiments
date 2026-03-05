#!/bin/bash
#SBATCH --job-name=toposeed_resnet
#SBATCH --output=outputs/logs/toposeed_resnet_%j.txt
#SBATCH --error=outputs/logs/toposeed_resnet_%j.err
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ---- Environment ------------------------------------------------------------
module purge
module load python/3.11
module load cuda/12.1     # adjust to your cluster

# Activate virtual environment (edit path as needed)
# source ~/venvs/topo/bin/activate

# ---- Run --------------------------------------------------------------------
cd "$SLURM_SUBMIT_DIR"

echo "=== TopoSeed ResNet-18 on CIFAR-10 ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
date

python examples/resnet_cifar10.py \
    --epochs 50 \
    --batch-size 128 \
    --lr 0.1 \
    --lambda-reg 0.005 \
    --device cuda

echo "Done."
date
