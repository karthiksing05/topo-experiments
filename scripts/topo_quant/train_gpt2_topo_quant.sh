#!/bin/bash
#SBATCH --job-name=gpt2_topo_quant
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=tail-lab
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --qos=long
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/gpt2_topo_quant-%j.out
#SBATCH --error=slurm/slurm_errors/gpt2_topo_quant-%j.err

# Train GPT-2 small (openai-community/gpt2 architecture, 124M params) on
# FineWeb-edu with SoftTopoQuantLoss applied to MLP layers (c_fc / c_proj).
#
# Reference quantised model: TheBloke/gpt2-GPTQ  (4-bit GPTQ)
# Our variant aims to learn weight distributions that are naturally more
# amenable to post-training 4-bit quantisation than the standard baseline.
#
# Configurable via sbatch --export=VAR=value:
#   VARIANT     'baseline' | 'topo_quant' | 'both'  (default: topo_quant)
#   MAX_STEPS   total training steps              (default from JSON: 100000)
#   TAU         TopoQuantLoss weight              (default from JSON: 1.0)
#   NUM_BITS    target quantisation bit-width     (default from JSON: 4)
#   BATCH_SIZE  per-GPU batch size                (default from JSON: 8)
#   DEVICE      cuda device string                (default: cuda:0)
#   PLOT_ONLY   set to 1 to skip training and just regenerate comparison plots

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
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/transformers"
export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

if [[ -f "${EXPERIMENT_DIR}/hf_token" ]]; then
    export HF_TOKEN=$(cat "${EXPERIMENT_DIR}/hf_token" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "=========================================================="
echo "GPT-2 small | TopoQuantLoss training on FineWeb-edu"
echo "  Reference: openai-community/gpt2 (+ TheBloke/gpt2-GPTQ)"
echo "=========================================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "=========================================================="

cd "$EXPERIMENT_DIR"
mkdir -p slurm/slurm_outputs slurm/slurm_errors
mkdir -p outputs/topo_quant_gpt2

# -- Configurable knobs --------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/topo_quant_gpt2.json}"
VARIANT="${VARIANT:-topo_quant}"
MAX_STEPS="${MAX_STEPS:-}"
DEVICE="${DEVICE:-cuda:0}"
PLOT_ONLY="${PLOT_ONLY:-0}"

echo ""
echo "config   : ${CONFIG_FILE}"
echo "variant  : ${VARIANT}"
echo "device   : ${DEVICE}"
[[ -n "$MAX_STEPS" ]] && echo "max_steps: ${MAX_STEPS}  (override)"
echo ""

OVERRIDE_ARGS="--variant ${VARIANT} --device ${DEVICE}"
[[ -n "$MAX_STEPS" ]] && OVERRIDE_ARGS+=" --max-steps ${MAX_STEPS}"
[[ "$PLOT_ONLY" == "1" ]] && OVERRIDE_ARGS+=" --plot-only"

# -- GPU diagnostics -----------------------------------------------------------
echo "--- GPU diagnostics ---"
nvidia-smi || echo "WARNING: nvidia-smi not found or no GPU visible"
python -c "import torch; print('torch.cuda.is_available():', torch.cuda.is_available()); \
           print('CUDA device count:', torch.cuda.device_count()); \
           [print(f'  GPU {i}:', torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]"
echo ""

# -- Train ---------------------------------------------------------------------
echo "--- Starting training ---"
srun python src/topo_quant/train_gpt2_topo_quant.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

echo ""
echo "--- Done ---"
echo "Checkpoints : outputs/topo_quant_gpt2/${VARIANT}/checkpoint_latest.pt"
echo "Log         : outputs/topo_quant_gpt2/${VARIANT}/log.json"
