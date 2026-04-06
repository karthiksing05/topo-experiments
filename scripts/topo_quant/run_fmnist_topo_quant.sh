#!/bin/bash
#SBATCH --job-name=fmnist_topo_quant
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=tail-lab
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/fmnist_topo_quant-%j.out
#SBATCH --error=slurm/slurm_errors/fmnist_topo_quant-%j.err

# Preliminary benchmark of TopoQuantLoss on FashionMNIST.
# Trains four variants:
#   baseline | topo_quant | soft_topo_quant | combined_topo_quant
#
# Outputs final test accuracy, parameter count, and model size (FP32 + 4-bit)
# to outputs/topo_quant_fmnist/.
#
# Configurable via sbatch --export=VAR=value:
#   EPOCHS     number of training epochs (default from JSON)
#   BATCH_SIZE batch size
#   LR         learning rate
#   TAU        TopoQuantLoss weight (default 1.0)
#   NUM_BITS   target quantisation bit-width (default 4)
#   DEVICE     cuda device string (default cuda:0)

set -euo pipefail

# -- Environment ---------------------------------------------------------------
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
export PYTHONIOENCODING=utf-8

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"
HF_CACHE_DIR="${EXPERIMENT_DIR}/.hf_cache"
export HF_HOME="${HF_CACHE_DIR}"
export HF_DATASETS_CACHE="${HF_CACHE_DIR}/datasets"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

HF_TOKEN_FILE="${EXPERIMENT_DIR}/wandb_token"
if [[ -f "${EXPERIMENT_DIR}/hf_token" ]]; then
    export HF_TOKEN=$(cat "${EXPERIMENT_DIR}/hf_token" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=========================================================="
echo "FashionMNIST TopoQuantLoss Preliminary Benchmark"
echo "  baseline | topo_quant | soft_topo_quant | combined_topo_quant"
echo "=========================================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd "$EXPERIMENT_DIR"
mkdir -p slurm/slurm_outputs slurm/slurm_errors
mkdir -p outputs/topo_quant_fmnist

# -- Configurable knobs --------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/topo_quant_fmnist.json}"
EPOCHS="${EPOCHS:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
TAU="${TAU:-}"
NUM_BITS="${NUM_BITS:-}"
DEVICE="${DEVICE:-cuda:0}"

echo ""
echo "config   : ${CONFIG_FILE}"
echo "device   : ${DEVICE}"
[[ -n "$EPOCHS"     ]] && echo "epochs   : ${EPOCHS}     (override)"
[[ -n "$TAU"        ]] && echo "tau      : ${TAU}        (override)"
[[ -n "$NUM_BITS"   ]] && echo "num_bits : ${NUM_BITS}   (override)"
echo ""

OVERRIDE_ARGS="--device ${DEVICE}"
[[ -n "$EPOCHS"     ]] && OVERRIDE_ARGS+=" --epochs ${EPOCHS}"
[[ -n "$NUM_BITS"   ]] && OVERRIDE_ARGS+=" --num-bits ${NUM_BITS}"
[[ -n "$TAU"        ]] && OVERRIDE_ARGS+=" --tau ${TAU}"

# -- GPU diagnostics -----------------------------------------------------------
echo "--- GPU diagnostics ---"
nvidia-smi || echo "WARNING: nvidia-smi not found or no GPU visible"
python -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); \
           print('CUDA device count:', torch.cuda.device_count()); \
           [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
echo ""

# -- Run -----------------------------------------------------------------------
echo "--- Running benchmark ---"
srun python src/topo_quant/fmnist_topo_quant.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "--- Done ---"
echo "Results: outputs/topo_quant_fmnist/results_latest.json"
