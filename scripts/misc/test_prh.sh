#!/bin/bash
#SBATCH --job-name=test_prh
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH -t 4:00:00
#SBATCH -qinferno
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/test_prh-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/test_prh-%j.err

# -- Environment ---------------------------------------------------------------
module purge
source /storage/home/hcoda1/3/ksingara3/scratch/miniconda3/etc/profile.d/conda.sh
conda deactivate
conda activate topovlm

EXPERIMENT_DIR="/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments"
HF_CACHE_DIR="${EXPERIMENT_DIR}/huggingfacehub_cache"

export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

# HuggingFace authentication token
HF_TOKEN_FILE="${EXPERIMENT_DIR}/litcoder_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE"
fi

echo "=========================================="
echo "topo-experiments -- PRH Activation Test"
echo "=========================================="
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURM_NODELIST"
echo "GPUs            : $CUDA_VISIBLE_DEVICES"
echo "HF cache        : $HF_CACHE_DIR"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs
mkdir -p "${HF_CACHE_DIR}"
mkdir -p outputs/test_prh

echo ""
srun python -u src/test/test_prh.py
echo ""

echo "=========================================="
echo "Test complete.  Outputs in:"
echo "  outputs/test_prh/"
echo "=========================================="
