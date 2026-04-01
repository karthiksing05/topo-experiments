#!/bin/bash
#SBATCH --job-name=split_fmnist
#SBATCH --account=tail-lab
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/split_fmnist-%j.out
#SBATCH --error=slurm/slurm_errors/split_fmnist-%j.err

# -- Environment ---------------------------------------------------------------
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"
HF_CACHE_DIR="${EXPERIMENT_DIR}/huggingfacehub_cache"

export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

HF_TOKEN_FILE="${EXPERIMENT_DIR}/litcoder_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=========================================="
echo "Split-FashionMNIST Continual Learning Experiment"
echo "  baseline | topo_only | topo_sparsity | topo_auxk_pooled | topo_regionlock"
echo "  5 sequential binary tasks — class-incremental, single 10-way head"
echo "=========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p slurm/slurm_outputs slurm/slurm_errors
mkdir -p outputs/split_fmnist/checkpoints
mkdir -p outputs/split_fmnist/results
mkdir -p outputs/split_fmnist/figures

# -- Configurable via sbatch --export= -----------------------------------------
# TASK_EPOCHS : epochs per task (default from JSON config)
# BATCH_SIZE  : batch size
# LR          : learning rate
# DEVICE      : cuda device string

CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/split_fmnist.json}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
TASK_EPOCHS="${TASK_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
DEVICE="${DEVICE:-cuda:0}"

echo ""
echo "config     : ${CONFIG_FILE}"
echo "device     : ${DEVICE}"
[[ -n "$TASK_EPOCHS" ]] && echo "task_epochs: ${TASK_EPOCHS} (override)"
[[ -n "$LR"          ]] && echo "lr         : ${LR}          (override)"
echo ""

OVERRIDE_ARGS=""
[[ -n "$DATA_DIR"    ]] && OVERRIDE_ARGS+=" --data-dir ${DATA_DIR}"
[[ -n "$OUTPUT_DIR"  ]] && OVERRIDE_ARGS+=" --output-dir ${OUTPUT_DIR}"
[[ -n "$TASK_EPOCHS" ]] && OVERRIDE_ARGS+=" --task-epochs ${TASK_EPOCHS}"
[[ -n "$BATCH_SIZE"  ]] && OVERRIDE_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LR"          ]] && OVERRIDE_ARGS+=" --lr ${LR}"
[[ -n "$DEVICE"      ]] && OVERRIDE_ARGS+=" --device ${DEVICE}"

# -- Phase 1: Run experiment ---------------------------------------------------
echo "--- Running experiment ---"
srun python -u src/fashion_mnist_forgetting/split_fmnist_forgetting.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "=========================================="
echo "Experiment complete.  Results in:"
echo "  outputs/split_fmnist/results/"
echo "=========================================="

# -- Phase 2: Generate analysis figures ----------------------------------------
echo ""
echo "--- Generating analysis figures ---"
python src/fashion_mnist_forgetting/analyze_split_fmnist.py \
    --results   outputs/split_fmnist/results/split_fmnist_results_latest.json \
    --out-dir   outputs/split_fmnist/figures \
    --ckpt-dir  outputs/split_fmnist/checkpoints \
    --device    "${DEVICE:-cuda:0}" \
    --strict

echo ""
echo "=========================================="
echo "Analysis complete.  Figures in:"
echo "  outputs/split_fmnist/figures/"
ls -lh outputs/split_fmnist/figures/ 2>/dev/null || true
echo "=========================================="
echo "Job finished at $(date)"
