#!/bin/bash
#SBATCH --job-name=resnet_topo_sparsity
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=12
#SBATCH -t 3-00:00:00
#SBATCH -qinferno
#SBATCH --gres=gpu:A100:2
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/train_resnet_topo_sparsity-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/train_resnet_topo_sparsity-%j.err

# ResNet-18 + TopoLoss + entropy-sparsity penalty on ImageNet (FFCV).
# Requires FFCV .beton files — run write_imagenet_ffcv.sh first.

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
export CUPY_CACHE_DIR="${EXPERIMENT_DIR}/.cupy_cache"

HF_TOKEN_FILE="${EXPERIMENT_DIR}/litcoder_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE"
fi

echo "=========================================="
echo "  ResNet-18: TopoLoss + Entropy-Sparsity (FFCV)"
echo "=========================================="
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURM_NODELIST"
echo "GPUs     : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs outputs/train_resnet_imagenet/checkpoints

CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/train_resnet_imagenet.json}"
TRAIN_BETON="${TRAIN_BETON:-${EXPERIMENT_DIR}/data/imagenet_ffcv/train.beton}"
VAL_BETON="${VAL_BETON:-${EXPERIMENT_DIR}/data/imagenet_ffcv/val.beton}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
MOMENTUM="${MOMENTUM:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-12}"
PRINT_FREQ="${PRINT_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
INCLUDE_DOWNSAMPLE="${INCLUDE_DOWNSAMPLE:-}"
RESUME="${RESUME:-}"

echo "config       : ${CONFIG_FILE}"
echo "train_beton  : ${TRAIN_BETON}"
echo "val_beton    : ${VAL_BETON}"
echo "device       : ${DEVICE}"
echo ""

OVERRIDE_ARGS=""
[[ -n "$OUTPUT_DIR"   ]] && OVERRIDE_ARGS+=" --output-dir ${OUTPUT_DIR}"
[[ -n "$EPOCHS"       ]] && OVERRIDE_ARGS+=" --epochs ${EPOCHS}"
[[ -n "$BATCH_SIZE"   ]] && OVERRIDE_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LR"           ]] && OVERRIDE_ARGS+=" --lr ${LR}"
[[ -n "$MOMENTUM"     ]] && OVERRIDE_ARGS+=" --momentum ${MOMENTUM}"
[[ -n "$WEIGHT_DECAY" ]] && OVERRIDE_ARGS+=" --weight-decay ${WEIGHT_DECAY}"
[[ -n "$NUM_WORKERS"  ]] && OVERRIDE_ARGS+=" --num-workers ${NUM_WORKERS}"
[[ -n "$PRINT_FREQ"   ]] && OVERRIDE_ARGS+=" --print-freq ${PRINT_FREQ}"
[[ -n "$SAVE_FREQ"    ]] && OVERRIDE_ARGS+=" --save-freq ${SAVE_FREQ}"
[[ -n "$RESUME"       ]] && OVERRIDE_ARGS+=" --resume ${RESUME}"

if [[ "${INCLUDE_DOWNSAMPLE}" == "1" || "${INCLUDE_DOWNSAMPLE}" == "true" ]]; then
    OVERRIDE_ARGS+=" --include-downsample"
fi

srun python -u src/imagenet/train_resnet_topo_sparsity.py \
    --config      "${CONFIG_FILE}" \
    --train-beton "${TRAIN_BETON}" \
    --val-beton   "${VAL_BETON}" \
    --device      "${DEVICE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "=========================================="
echo "Training complete. Checkpoints in:"
echo "  outputs/train_resnet_imagenet/checkpoints/best_topo_sparsity.pt"
echo "=========================================="
