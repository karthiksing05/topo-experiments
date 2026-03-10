#!/bin/bash
#SBATCH --job-name=write_imagenet_ffcv
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=16
#SBATCH -t 12:00:00
#SBATCH -qinferno
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/write_imagenet_ffcv-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/write_imagenet_ffcv-%j.err

# Converts ILSVRC/imagenet-1k from HuggingFace Hub to FFCV .beton format.
# Run this ONCE before submitting any training jobs.
# Output: data/imagenet_ffcv/train.beton  (~120 GB)
#         data/imagenet_ffcv/val.beton    (~  6 GB)

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

HF_TOKEN_FILE="${EXPERIMENT_DIR}/litcoder_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE"
fi

echo "=========================================="
echo "  Writing ImageNet-1k -> FFCV .beton"
echo "=========================================="
echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $SLURM_NODELIST"
echo "HF cache    : $HF_CACHE_DIR"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs data/imagenet_ffcv

TRAIN_OUT="${TRAIN_OUT:-${EXPERIMENT_DIR}/data/imagenet_ffcv/train.beton}"
VAL_OUT="${VAL_OUT:-${EXPERIMENT_DIR}/data/imagenet_ffcv/val.beton}"
NUM_WORKERS="${NUM_WORKERS:-16}"
MAX_RESOLUTION="${MAX_RESOLUTION:-500}"
JPEG_QUALITY="${JPEG_QUALITY:-90}"

echo "train_out       : ${TRAIN_OUT}"
echo "val_out         : ${VAL_OUT}"
echo "num_workers     : ${NUM_WORKERS}"
echo "max_resolution  : ${MAX_RESOLUTION}"
echo "jpeg_quality    : ${JPEG_QUALITY}"
echo ""

srun python -u src/imagenet/write_imagenet_ffcv.py \
    --train-out      "${TRAIN_OUT}" \
    --val-out        "${VAL_OUT}" \
    --num-workers    "${NUM_WORKERS}" \
    --max-resolution "${MAX_RESOLUTION}" \
    --jpeg-quality   "${JPEG_QUALITY}" \
    --hf-token       "${HF_TOKEN}"

echo ""
echo "=========================================="
echo "FFCV write complete."
echo "  train : ${TRAIN_OUT}"
echo "  val   : ${VAL_OUT}"
echo "=========================================="
