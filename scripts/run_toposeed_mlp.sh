#!/bin/bash
#SBATCH --job-name=toposeed_mlp
#SBATCH --output=outputs/logs/toposeed_mlp_%j.txt
#SBATCH --error=outputs/logs/toposeed_mlp_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

echo "=== TopoSeed MLP on Fashion-MNIST ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $SLURM_NODELIST"
date

python examples/mlp_fashion_mnist.py \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --lambda-cross 0.005 \
    --device cuda

echo "Done."
date
