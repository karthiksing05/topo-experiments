#!/bin/bash
#SBATCH --job-name=eval_toxicity_nanogpt_quant
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/eval_toxicity_nanogpt_quant-%j.out
#SBATCH --error=slurm/slurm_errors/eval_toxicity_nanogpt_quant-%j.err

# Evaluate the topo toxicity benchmark with quantized nanoGPT checkpoints.
#
# This script runs the same experiments as eval_toxicity_nanogpt.sh but with
# post-training quantization applied to every checkpoint before evaluation.
# Results are saved to outputs/toxicity_nanogpt_quantized/.
#
# The quantized forward pass runs natively on GPU:
#   fp16  — float16 weights, CUDA Tensor Cores (default; any GPU)
#   bf16  — bfloat16 weights, wider dynamic range (recommended on A100+)
#   int8  — bitsandbytes 8-bit GEMM (requires bitsandbytes; falls back to fp16)
#
# Configurable knobs (override via: sbatch --export=VAR=value ...):
#   QUANTIZATION          quantization mode: fp16 | bf16 | int8 (default fp16)
#   N_PROMPTS             number of toxic prompts from RealToxicityPrompts (default 200)
#   N_GEN                 number of completions per prompt (default 1)
#   MAX_NEW_TOKENS        new tokens to generate per completion (default 200)
#   TEMPERATURE           sampling temperature (default 1.0)
#   TOP_K                 top-k sampling cutoff; 0 = greedy (default 50)
#   TAUS                  comma-separated tau values (default "0.0,0.5,1.0,3.0,50.0")
#   PRUNING_FRACS         comma-separated pruning fractions (default "0.05,0.1,0.15,0.2")
#   N_SELECTIVITY_TOKENS  token budget for activation collection (default 4096)
#   NO_PRUNING            set to 1 to skip the pruning sweep (faster)
#   SVD_PRUNING_FRACS     comma-separated SVD pruning fractions (default "0.05,0.1,0.15,0.2")
#   NO_SVD_PRUNING        set to 1 to skip SVD pruning sweep
#   AMP_FACTOR            amplification factor for toxic neurons (default 5.0)
#   AMP_FRACS             comma-separated fractions for amplification sweep (default "0.05,0.1,0.15,0.2")
#   NO_AMPLIFICATION      set to 1 to skip the amplification sweep
#
# Examples:
#   # Run with default fp16 quantization
#   sbatch scripts/misc/eval_toxicity_nanogpt_quantized.sh
#
#   # Run with bfloat16 on a subset of taus, no pruning
#   sbatch --export=QUANTIZATION=bf16,TAUS="0.0,1.0,50.0",NO_PRUNING=1 \
#          scripts/misc/eval_toxicity_nanogpt_quantized.sh
#
#   # Run int8 on more prompts
#   sbatch --export=QUANTIZATION=int8,N_PROMPTS=500 \
#          scripts/misc/eval_toxicity_nanogpt_quantized.sh

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

# HuggingFace authentication token (needed for gated datasets)
HF_TOKEN_FILE="${EXPERIMENT_DIR}/hf_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE"
fi

# -- Configurable knobs --------------------------------------------------------
QUANTIZATION="${QUANTIZATION:-fp16}"
N_PROMPTS="${N_PROMPTS:-200}"
N_GEN="${N_GEN:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
TAUS="${TAUS:-3.0,50.0,0.0,0.5,1.0}"
PRUNING_FRACS="${PRUNING_FRACS:-0.05,0.1,0.15,0.2}"
N_SELECTIVITY_TOKENS="${N_SELECTIVITY_TOKENS:-4096}"
NO_PRUNING="${NO_PRUNING:-0}"
NO_SVD_PRUNING="${NO_SVD_PRUNING:-0}"
AMP_FACTOR="${AMP_FACTOR:-5.0}"
AMP_FRACS="${AMP_FRACS:-0.05,0.1,0.15,0.2}"
NO_AMPLIFICATION="${NO_AMPLIFICATION:-0}"

# -- Job info ------------------------------------------------------------------
OUTPUT_SUBDIR="outputs/toxicity_nanogpt_quantized_${QUANTIZATION}"

echo "========================================="
echo "Topo NanoGPT -- Quantized Toxicity Benchmark"
echo "========================================="
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "HF cache        : $HF_CACHE_DIR"
echo "quantization    : $QUANTIZATION"
echo "output dir      : $OUTPUT_SUBDIR"
echo "n_prompts       : $N_PROMPTS"
echo "n_gen           : $N_GEN"
echo "max_new_tokens  : $MAX_NEW_TOKENS"
echo "temperature     : $TEMPERATURE"
echo "top_k           : $TOP_K"
echo "taus            : $TAUS"
echo "pruning_fracs   : $PRUNING_FRACS"
echo "selectivity_toks: $N_SELECTIVITY_TOKENS"
echo "no_pruning      : $NO_PRUNING"
echo "no_svd_pruning  : $NO_SVD_PRUNING"
echo "amp_factor      : $AMP_FACTOR"
echo "amp_fracs       : $AMP_FRACS"
echo "no_amplification: $NO_AMPLIFICATION"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${OUTPUT_SUBDIR}"

srun python -u src/test/eval_toxicity_nanogpt_quantized.py \
    --quantization   "$QUANTIZATION"   \
    --n_prompts      "$N_PROMPTS"      \
    --n_gen          "$N_GEN"          \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature    "$TEMPERATURE"    \
    --top_k          "$TOP_K"          \
    --taus           "$TAUS"           \
    --pruning_fracs  "$PRUNING_FRACS"  \
    --n_selectivity_tokens "$N_SELECTIVITY_TOKENS" \
    $( [[ "$NO_PRUNING"       == "1" ]] && echo "--no_pruning" ) \
    $( [[ "$NO_SVD_PRUNING"   == "1" ]] && echo "--no_svd_pruning" ) \
    --amp_factor     "$AMP_FACTOR" \
    --amp_fracs      "$AMP_FRACS" \
    $( [[ "$NO_AMPLIFICATION" == "1" ]] && echo "--no_amplification" )

echo "=========================================="
echo "Done. Results in: ${OUTPUT_SUBDIR}/"
echo "=========================================="

# =============================================================================
# Copy-paste commands to launch all four quantization modes:
#
#   sbatch                                   scripts/misc/eval_toxicity_nanogpt_quantized.sh
#   sbatch --export=ALL,QUANTIZATION=bf16    scripts/misc/eval_toxicity_nanogpt_quantized.sh
#   sbatch --export=ALL,QUANTIZATION=int8    scripts/misc/eval_toxicity_nanogpt_quantized.sh
#   sbatch --export=ALL,QUANTIZATION=int4    scripts/misc/eval_toxicity_nanogpt_quantized.sh
#
# Outputs land in separate directories:
#   outputs/toxicity_nanogpt_quantized_fp16/
#   outputs/toxicity_nanogpt_quantized_bf16/
#   outputs/toxicity_nanogpt_quantized_int8/
#   outputs/toxicity_nanogpt_quantized_int4/
#
# int8 and int4 require bitsandbytes (pip install bitsandbytes).
# Both fall back to fp16 silently if the package is not installed.
# int4 uses NF4 + double-quantization (QLoRA); ~8x memory vs float32.
# =============================================================================
