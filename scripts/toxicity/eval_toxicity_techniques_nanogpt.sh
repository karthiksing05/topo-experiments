#!/bin/bash
#SBATCH --job-name=eval_tox_techniques
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=tail-lab
#SBATCH --partition=tail-lab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm_outputs/eval_tox_techniques-%j.out
#SBATCH --error=slurm/slurm_errors/eval_tox_techniques-%j.err

# Focused technique comparison for topo-nanoGPT detoxification:
#   per-layer pruning, global pruning,
#   per-layer DAA, global DAA,
#   per-layer OSD, global OSD
# at fractions [20%, 50%] on:
#   RealToxicityPrompts  (allenai/real-toxicity-prompts)
#   ToxiGen              (microsoft/toxigen)
# scored by Detoxify (always) and optionally Perspective API.
#
# Configurable knobs (override via: sbatch --export=ALL,VAR=value ...):
#
#   TAUS                  comma-separated tau values  (default "0.0,0.5,1.0,3.0,50.0")
#   FRACS                 comma-separated intervention fractions  (default "0.2,0.5")
#   N_PROMPTS             toxic prompts per dataset  (default 200)
#   N_TOXIGEN_PROMPTS     ToxiGen prompts; falls back to N_PROMPTS if unset
#   N_GEN                 completions per prompt  (default 1)
#   MAX_NEW_TOKENS        new tokens per completion  (default 200)
#   TEMPERATURE           sampling temperature  (default 1.0)
#   TOP_K                 top-k cutoff; 0 = greedy  (default 50)
#   N_SELECTIVITY_TOKENS  activation-collection token budget  (default 4096)
#   N_OSD_COMPONENTS      toxic PCA components per layer  (default 32)
#   N_CLEAN_COMPONENTS    clean PCA components for OSD  (default 32)
#   NO_PRUNING            set to 1 to skip neuron pruning methods
#   NO_DAA                set to 1 to skip DAA methods
#   NO_OSD                set to 1 to skip OSD methods
#   NO_TOXIGEN            set to 1 to skip ToxiGen evaluation
#   NO_RTP                set to 1 to skip RealToxicityPrompts evaluation
#   PERSPECTIVE_API_KEY   Perspective API key (optional; skip if empty)
#   RESUME                set to 1 to skip taus whose JSON already exists
#   OUTPUT_DIR            output directory
#
# Examples
# --------
# Quick sanity check (no pruning, RTP only, 50 prompts, tau=0.0 only):
#   sbatch --export=ALL,N_PROMPTS=50,TAUS="0.0",NO_PRUNING=1,NO_TOXIGEN=1 \
#          scripts/toxicity/eval_toxicity_techniques_nanogpt.sh
#
# Full run with Perspective API:
#   sbatch --export=ALL,PERSPECTIVE_API_KEY=<key> \
#          scripts/toxicity/eval_toxicity_techniques_nanogpt.sh
#
# Resume interrupted run:
#   sbatch --export=ALL,RESUME=1 \
#          scripts/toxicity/eval_toxicity_techniques_nanogpt.sh

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

# HuggingFace authentication token (needed for gated datasets like ToxiGen)
HF_TOKEN_FILE="${EXPERIMENT_DIR}/hf_token"
if [[ -f "$HF_TOKEN_FILE" ]]; then
    export HF_TOKEN=$(cat "$HF_TOKEN_FILE" | tr -d '[:space:]')
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    echo "HF token loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: HF token file not found at $HF_TOKEN_FILE"
fi

# -- Configurable knobs --------------------------------------------------------
TAUS="${TAUS:-0.0,0.5,1.0,3.0,50.0}"
FRACS="${FRACS:-0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5}"
N_PROMPTS="${N_PROMPTS:-200}"
N_TOXIGEN_PROMPTS="${N_TOXIGEN_PROMPTS:-$N_PROMPTS}"
N_GEN="${N_GEN:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
N_SELECTIVITY_TOKENS="${N_SELECTIVITY_TOKENS:-4096}"
N_OSD_COMPONENTS="${N_OSD_COMPONENTS:-32}"
N_CLEAN_COMPONENTS="${N_CLEAN_COMPONENTS:-32}"
NO_PRUNING="${NO_PRUNING:-0}"
NO_DAA="${NO_DAA:-0}"
NO_OSD="${NO_OSD:-0}"
NO_TOXIGEN="${NO_TOXIGEN:-0}"
NO_RTP="${NO_RTP:-0}"
PERSPECTIVE_API_KEY="${PERSPECTIVE_API_KEY:-}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_techniques_nanogpt}"

# -- Job info ------------------------------------------------------------------
echo "=============================================="
echo "Topo NanoGPT -- Technique Comparison Eval"
echo "=============================================="
echo "Job ID              : ${SLURM_JOB_ID:-local}"
echo "Node                : ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "HF cache            : $HF_CACHE_DIR"
echo "taus                : $TAUS"
echo "fracs               : $FRACS"
echo "n_prompts (RTP)     : $N_PROMPTS"
echo "n_prompts (ToxiGen) : $N_TOXIGEN_PROMPTS"
echo "n_gen               : $N_GEN"
echo "max_new_tokens      : $MAX_NEW_TOKENS"
echo "temperature         : $TEMPERATURE"
echo "top_k               : $TOP_K"
echo "n_selectivity_tokens: $N_SELECTIVITY_TOKENS"
echo "n_osd_components    : $N_OSD_COMPONENTS"
echo "n_clean_components  : $N_CLEAN_COMPONENTS"
echo "no_pruning          : $NO_PRUNING"
echo "no_daa              : $NO_DAA"
echo "no_osd              : $NO_OSD"
echo "no_toxigen          : $NO_TOXIGEN"
echo "no_rtp              : $NO_RTP"
echo "perspective_api_key : $( [[ -n "$PERSPECTIVE_API_KEY" ]] && echo '(set)' || echo '(not set)' )"
echo "resume              : $RESUME"
echo "output_dir          : $OUTPUT_DIR"
echo "=============================================="

cd "$EXPERIMENT_DIR"
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p slurm/slurm_outputs
mkdir -p slurm/slurm_errors

# Pass Perspective API key only if set
PERSP_ARGS=()
if [[ -n "$PERSPECTIVE_API_KEY" ]]; then
    PERSP_ARGS+=(--perspective_key "$PERSPECTIVE_API_KEY")
fi

srun python -u src/toxicity/eval_toxicity_techniques_nanogpt.py \
    --taus                   "$TAUS"                  \
    --fracs                  "$FRACS"                 \
    --n_prompts              "$N_PROMPTS"             \
    --n_gen                  "$N_GEN"                 \
    --max_new_tokens         "$MAX_NEW_TOKENS"        \
    --temperature            "$TEMPERATURE"           \
    --top_k                  "$TOP_K"                 \
    --n_selectivity_tokens   "$N_SELECTIVITY_TOKENS"  \
    --n_osd_components       "$N_OSD_COMPONENTS"      \
    --n_clean_components     "$N_CLEAN_COMPONENTS"    \
    $( [[ "$NO_PRUNING"  == "1" ]] && echo "--no_pruning"  ) \
    $( [[ "$NO_DAA"      == "1" ]] && echo "--no_daa"      ) \
    $( [[ "$NO_OSD"      == "1" ]] && echo "--no_osd"      ) \
    $( [[ "$NO_TOXIGEN"  == "1" ]] && echo "--no_toxigen"  ) \
    $( [[ "$NO_RTP"      == "1" ]] && echo "--no_rtp"      ) \
    $( [[ "$RESUME"      == "1" ]] && echo "--resume"      ) \
    "${PERSP_ARGS[@]}" \
    --output_dir             "$OUTPUT_DIR"

echo "=============================================="
echo "Done. Results in: $OUTPUT_DIR/"
echo "=============================================="
