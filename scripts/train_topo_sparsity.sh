#!/bin/bash
#SBATCH --job-name=train_topo_sparsity
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH -t 8:00:00
#SBATCH -qinferno
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/train_topo_sparsity-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/train_topo_sparsity-%j.err

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
echo "topo-experiments -- Topo Sparsity vs Baseline (FashionMNIST CNN, fc2 excluded from reg)"
echo "=========================================="
echo "Job ID          : $SLURM_JOB_ID"
echo "Node            : $SLURM_NODELIST"
echo "GPUs            : $CUDA_VISIBLE_DEVICES"
echo "HF cache        : $HF_CACHE_DIR"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs
mkdir -p "${HF_CACHE_DIR}"
mkdir -p outputs/train_topo_sparsity/checkpoints

# -- Configurable via sbatch --export= -----------------------------------------
# Per-layer settings (topo_scale, factor_h, factor_w, lambda_kl, lambda_entropy)
# live in the JSON config file — edit configs/train_topo_sparsity.json directly.
# Top-level knobs below can still be overridden here or via --export on sbatch.

CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/train_topo_sparsity.json}"
DATA_DIR="${DATA_DIR:-}"                  # null → project-root/data (set in JSON or here)
OUTPUT_DIR="${OUTPUT_DIR:-}"              # null → project-root/outputs/train_topo_sparsity
EPOCHS="${EPOCHS:-}"                      # null → use JSON value
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
DEVICE="${DEVICE:-cuda:0}"
RESUME_TOPO="${RESUME_TOPO:-}"
RESUME_TOPO_ONLY="${RESUME_TOPO_ONLY:-}"
RESUME_BASE="${RESUME_BASE:-}"

echo ""
echo "config       : ${CONFIG_FILE}"
echo "device       : ${DEVICE}"
[[ -n "$DATA_DIR"   ]] && echo "data_dir     : ${DATA_DIR}   (CLI override)"
[[ -n "$EPOCHS"     ]] && echo "epochs       : ${EPOCHS}     (CLI override)"
[[ -n "$BATCH_SIZE" ]] && echo "batch_size   : ${BATCH_SIZE} (CLI override)"
[[ -n "$LR"         ]] && echo "lr           : ${LR}         (CLI override)"
echo ""

# Build optional CLI override args (only passed when env var is non-empty)
OVERRIDE_ARGS=""
[[ -n "$DATA_DIR"       ]] && OVERRIDE_ARGS+=" --data-dir ${DATA_DIR}"
[[ -n "$OUTPUT_DIR"     ]] && OVERRIDE_ARGS+=" --output-dir ${OUTPUT_DIR}"
[[ -n "$EPOCHS"         ]] && OVERRIDE_ARGS+=" --epochs ${EPOCHS}"
[[ -n "$BATCH_SIZE"     ]] && OVERRIDE_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LR"             ]] && OVERRIDE_ARGS+=" --lr ${LR}"
[[ -n "$DEVICE"         ]] && OVERRIDE_ARGS+=" --device ${DEVICE}"
[[ -n "$RESUME_TOPO"    ]] && OVERRIDE_ARGS+=" --resume-topo ${RESUME_TOPO}"
[[ -n "$RESUME_TOPO_ONLY" ]] && OVERRIDE_ARGS+=" --resume-topo-only ${RESUME_TOPO_ONLY}"
[[ -n "$RESUME_BASE"    ]] && OVERRIDE_ARGS+=" --resume-base ${RESUME_BASE}"

srun python -u src/train/train_topo_sparsity.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "=========================================="
echo "Training complete.  Outputs in:"
echo "  outputs/train_topo_sparsity/"
echo "=========================================="
