#!/bin/bash
#SBATCH --job-name=train_resnet_imagenet
#SBATCH --account=gts-aivanova7-lab
#SBATCH -p p-long
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=128GB
#SBATCH --cpus-per-task=8
#SBATCH -t 7-00:00:00
#SBATCH -qinferno
#SBATCH --gres=gpu:A100:2
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/train_resnet_imagenet-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/train_resnet_imagenet-%j.err

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

# HuggingFace authentication token (used for ImageNet-1k gated dataset download)
HF_TOKEN_FILE="${EXPERIMENT_DIR}/litcoder_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE"
fi

echo "=========================================="
echo "topo-experiments -- ResNet-18 TopoSparsity vs Baseline (ImageNet-1k)"
echo "  topo      : TopoLoss + entropy-sparsity on all residual-block convs"
echo "  topo_only : TopoLoss only (entropy penalty zeroed)"
echo "  baseline  : CrossEntropy only"
echo "=========================================="
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURM_NODELIST"
echo "GPUs            : $CUDA_VISIBLE_DEVICES"
echo "HF cache        : $HF_CACHE_DIR"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs
mkdir -p "${HF_CACHE_DIR}"
mkdir -p outputs/train_resnet_imagenet/checkpoints

# -- Configurable via sbatch --export= -----------------------------------------
# Per-layer TopoLoss / entropy settings live in the JSON config file.
# Edit configs/train_resnet_imagenet.json directly.
# Top-level knobs below can be overridden via --export on sbatch, e.g.:
#   sbatch --export=ALL,EPOCHS=10,BATCH_SIZE=128 train_resnet_imagenet.sh

CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/train_resnet_imagenet.json}"
DATA_DIR="${DATA_DIR:-}"           # null → project-root/data/imagenet (set in JSON or here)
OUTPUT_DIR="${OUTPUT_DIR:-}"       # null → project-root/outputs/train_resnet_imagenet
NUM_CLASSES="${NUM_CLASSES:-}"     # null → 1000 (full ImageNet)
EPOCHS="${EPOCHS:-}"               # null → use JSON value (90)
BATCH_SIZE="${BATCH_SIZE:-}"       # null → use JSON value (256)
LR="${LR:-}"                       # null → use JSON value (0.1)
MOMENTUM="${MOMENTUM:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-}"
DEVICE="${DEVICE:-cuda:0}"
NUM_WORKERS="${NUM_WORKERS:-}"     # null → use JSON value (8)
PRINT_FREQ="${PRINT_FREQ:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
INCLUDE_DOWNSAMPLE="${INCLUDE_DOWNSAMPLE:-}"   # set to "1"/"true" to enable
RESUME_TOPO="${RESUME_TOPO:-}"
RESUME_TOPO_ONLY="${RESUME_TOPO_ONLY:-}"
RESUME_BASE="${RESUME_BASE:-}"

echo ""
echo "config              : ${CONFIG_FILE}"
echo "device              : ${DEVICE}"
[[ -n "$DATA_DIR"    ]] && echo "data_dir            : ${DATA_DIR}    (CLI override)"
[[ -n "$NUM_CLASSES" ]] && echo "num_classes         : ${NUM_CLASSES} (CLI override)"
[[ -n "$EPOCHS"      ]] && echo "epochs              : ${EPOCHS}      (CLI override)"
[[ -n "$BATCH_SIZE"  ]] && echo "batch_size          : ${BATCH_SIZE}  (CLI override)"
[[ -n "$LR"          ]] && echo "lr                  : ${LR}          (CLI override)"
[[ -n "$MOMENTUM"    ]] && echo "momentum            : ${MOMENTUM}    (CLI override)"
[[ -n "$WEIGHT_DECAY" ]] && echo "weight_decay        : ${WEIGHT_DECAY} (CLI override)"
[[ -n "$NUM_WORKERS" ]] && echo "num_workers         : ${NUM_WORKERS} (CLI override)"
echo ""

# Build optional CLI override args (only passed when env var is non-empty)
OVERRIDE_ARGS=""
[[ -n "$DATA_DIR"         ]] && OVERRIDE_ARGS+=" --data-dir ${DATA_DIR}"
[[ -n "$OUTPUT_DIR"       ]] && OVERRIDE_ARGS+=" --output-dir ${OUTPUT_DIR}"
[[ -n "$NUM_CLASSES"      ]] && OVERRIDE_ARGS+=" --num-classes ${NUM_CLASSES}"
[[ -n "$EPOCHS"           ]] && OVERRIDE_ARGS+=" --epochs ${EPOCHS}"
[[ -n "$BATCH_SIZE"       ]] && OVERRIDE_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LR"               ]] && OVERRIDE_ARGS+=" --lr ${LR}"
[[ -n "$MOMENTUM"         ]] && OVERRIDE_ARGS+=" --momentum ${MOMENTUM}"
[[ -n "$WEIGHT_DECAY"     ]] && OVERRIDE_ARGS+=" --weight-decay ${WEIGHT_DECAY}"
[[ -n "$DEVICE"           ]] && OVERRIDE_ARGS+=" --device ${DEVICE}"
[[ -n "$NUM_WORKERS"      ]] && OVERRIDE_ARGS+=" --num-workers ${NUM_WORKERS}"
[[ -n "$PRINT_FREQ"       ]] && OVERRIDE_ARGS+=" --print-freq ${PRINT_FREQ}"
[[ -n "$SAVE_FREQ"        ]] && OVERRIDE_ARGS+=" --save-freq ${SAVE_FREQ}"
[[ -n "$RESUME_TOPO"      ]] && OVERRIDE_ARGS+=" --resume-topo ${RESUME_TOPO}"
[[ -n "$RESUME_TOPO_ONLY" ]] && OVERRIDE_ARGS+=" --resume-topo-only ${RESUME_TOPO_ONLY}"
[[ -n "$RESUME_BASE"      ]] && OVERRIDE_ARGS+=" --resume-base ${RESUME_BASE}"

# --include-downsample flag (opt-in)
if [[ "${INCLUDE_DOWNSAMPLE}" == "1" || "${INCLUDE_DOWNSAMPLE}" == "true" || "${INCLUDE_DOWNSAMPLE}" == "yes" ]]; then
    OVERRIDE_ARGS+=" --include-downsample"
fi

# Pass HF token explicitly so the Python script can download ImageNet-1k if needed
[[ -n "$HF_TOKEN" ]] && OVERRIDE_ARGS+=" --hf-token ${HF_TOKEN}"

srun python -u src/imagenet/train_resnet_imagenet.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "=========================================="
echo "Training complete.  Outputs in:"
echo "  outputs/train_resnet_imagenet/"
echo "=========================================="
