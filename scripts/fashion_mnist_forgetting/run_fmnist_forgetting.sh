#!/bin/bash
#SBATCH --job-name=fmnist_forgetting
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH -t 4:00:00
#SBATCH -q inferno
#SBATCH --gres=gpu:1
#SBATCH --output=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/fmnist_forgetting-%j.out
#SBATCH --error=/storage/home/hcoda1/3/ksingara3/scratch/topo-experiments/scripts/logs/fmnist_forgetting-%j.err

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

HF_TOKEN_FILE="${EXPERIMENT_DIR}/litcoder_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=========================================="
echo "FashionMNIST Catastrophic Forgetting Experiment"
echo "  baseline | topo_only | topo_sparsity | topo_auxk"
echo "=========================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p scripts/logs
mkdir -p outputs/fashion_mnist_forgetting/checkpoints
mkdir -p outputs/fashion_mnist_forgetting/results

# -- Configurable via sbatch --export= -----------------------------------------
# noise_target_class: which FashionMNIST class to use as the finetune target label
#   0=T-shirt  1=Trouser  2=Pullover  3=Dress  4=Coat
#   5=Sandal   6=Shirt    7=Sneaker   8=Bag    9=AnkleBoot
# ft_source_class: source class whose real images are used as stimuli (relabeled
#   as noise_target_class). Leave unset for pure-noise mode.

CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/fashion_mnist_forgetting.json}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
FINETUNE_LR="${FINETUNE_LR:-}"
DEVICE="${DEVICE:-cuda:0}"
NOISE_TARGET="${NOISE_TARGET:-}"
NOISE_SAMPLES="${NOISE_SAMPLES:-}"
FT_SOURCE_CLASS="${FT_SOURCE_CLASS:-}"

echo ""
echo "config          : ${CONFIG_FILE}"
echo "device          : ${DEVICE}"
[[ -n "$PRETRAIN_EPOCHS" ]] && echo "pretrain_epochs : ${PRETRAIN_EPOCHS} (override)"
[[ -n "$FINETUNE_EPOCHS" ]] && echo "finetune_epochs : ${FINETUNE_EPOCHS} (override)"
[[ -n "$NOISE_TARGET"    ]] && echo "noise_target    : ${NOISE_TARGET}    (override)"
[[ -n "$FT_SOURCE_CLASS" ]] && echo "ft_source_class : ${FT_SOURCE_CLASS}  (override)"
echo ""

OVERRIDE_ARGS=""
[[ -n "$DATA_DIR"         ]] && OVERRIDE_ARGS+=" --data-dir ${DATA_DIR}"
[[ -n "$OUTPUT_DIR"       ]] && OVERRIDE_ARGS+=" --output-dir ${OUTPUT_DIR}"
[[ -n "$PRETRAIN_EPOCHS"  ]] && OVERRIDE_ARGS+=" --pretrain-epochs ${PRETRAIN_EPOCHS}"
[[ -n "$FINETUNE_EPOCHS"  ]] && OVERRIDE_ARGS+=" --finetune-epochs ${FINETUNE_EPOCHS}"
[[ -n "$BATCH_SIZE"       ]] && OVERRIDE_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LR"               ]] && OVERRIDE_ARGS+=" --lr ${LR}"
[[ -n "$FINETUNE_LR"      ]] && OVERRIDE_ARGS+=" --finetune-lr ${FINETUNE_LR}"
[[ -n "$DEVICE"           ]] && OVERRIDE_ARGS+=" --device ${DEVICE}"
[[ -n "$NOISE_TARGET"     ]] && OVERRIDE_ARGS+=" --noise-target-class ${NOISE_TARGET}"
[[ -n "$NOISE_SAMPLES"    ]] && OVERRIDE_ARGS+=" --noise-samples ${NOISE_SAMPLES}"
[[ -n "$FT_SOURCE_CLASS"  ]] && OVERRIDE_ARGS+=" --ft-source-class ${FT_SOURCE_CLASS}"

srun python -u src/fashion_mnist_forgetting/fmnist_forgetting.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "=========================================="
echo "Experiment complete.  Results in:"
echo "  outputs/fashion_mnist_forgetting/results/"
echo "=========================================="
