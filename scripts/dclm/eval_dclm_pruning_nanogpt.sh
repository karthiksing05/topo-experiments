#!/bin/bash
#SBATCH --job-name=eval_dclm_pruning
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/slurm_outputs/eval_dclm_pruning-%j.out
#SBATCH --error=slurm/slurm_errors/eval_dclm_pruning-%j.err

# Cross-task pruning experiment on topo-nanoGPT with DCLM-core tasks.
#
# For each task, develops selectivity (task-specific vs generic activations),
# applies 6 pruning strategies at multiple fractions, and evaluates on ALL
# tasks to produce a cross-task impact matrix.
#
# Tasks: hellaswag, piqa, arc_easy, arc_challenge, winogrande, boolq, openbookqa
# Methods: per-layer/global × pruning/DAA/OSD
#
# Configurable knobs (override via: sbatch --export=ALL,VAR=value ...):
#
#   TAUS                 comma-separated tau values (default "0.0,0.5,1.0,3.0,50.0")
#   FRACS                comma-separated fractions  (default "0.0,0.05,...,0.5")
#   N_EVAL               examples per task for accuracy evaluation (default 200)
#   N_SELECT             examples per task for activation collection (default 200)
#   N_GENERIC            generic OWT texts for contrast (default 200)
#   MAX_SELECT_TOKENS    max tokens per text during activation collection (default 64)
#   N_OSD_COMPONENTS     OSD components per layer (default 32)
#   N_CLEAN_COMPONENTS   clean PCA components for OSD (default 32)
#   TASKS                comma-separated tasks (default all 7)
#   RESUME               set to 1 to skip taus whose JSON already exists
#   OUTPUT_DIR           output directory
#
# Examples
# --------
# Quick test (1 tau, 2 tasks, 50 examples):
#   sbatch --export=ALL,TAUS="0.0",TASKS="hellaswag,piqa",N_EVAL=50,N_SELECT=50 \
#          scripts/dclm/eval_dclm_pruning_nanogpt.sh
#
# Full run:
#   sbatch scripts/dclm/eval_dclm_pruning_nanogpt.sh
#
# Resume after interruption:
#   sbatch --export=ALL,RESUME=1 scripts/dclm/eval_dclm_pruning_nanogpt.sh

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

# HuggingFace authentication token
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
N_EVAL="${N_EVAL:-200}"
N_SELECT="${N_SELECT:-200}"
N_GENERIC="${N_GENERIC:-200}"
MAX_SELECT_TOKENS="${MAX_SELECT_TOKENS:-64}"
N_OSD_COMPONENTS="${N_OSD_COMPONENTS:-32}"
N_CLEAN_COMPONENTS="${N_CLEAN_COMPONENTS:-32}"
TASKS="${TASKS:-hellaswag,piqa,arc_easy,arc_challenge,winogrande,boolq,openbookqa}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/dclm_pruning_nanogpt}"

# -- Job info ------------------------------------------------------------------
echo "=============================================="
echo "Topo NanoGPT -- DCLM Cross-Task Pruning"
echo "=============================================="
echo "Job ID              : ${SLURM_JOB_ID:-local}"
echo "Node                : ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "HF cache            : $HF_CACHE_DIR"
echo "taus                : $TAUS"
echo "fracs               : $FRACS"
echo "n_eval              : $N_EVAL"
echo "n_select            : $N_SELECT"
echo "n_generic           : $N_GENERIC"
echo "max_select_tokens   : $MAX_SELECT_TOKENS"
echo "n_osd_components    : $N_OSD_COMPONENTS"
echo "n_clean_components  : $N_CLEAN_COMPONENTS"
echo "tasks               : $TASKS"
echo "resume              : $RESUME"
echo "output_dir          : $OUTPUT_DIR"
echo "=============================================="

cd "$EXPERIMENT_DIR"
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p slurm/slurm_outputs
mkdir -p slurm/slurm_errors

srun python -u src/dclm/eval_dclm_pruning_nanogpt.py \
    --taus                 "$TAUS"               \
    --fracs                "$FRACS"              \
    --n_eval               "$N_EVAL"             \
    --n_select             "$N_SELECT"           \
    --n_generic            "$N_GENERIC"          \
    --max_select_tokens    "$MAX_SELECT_TOKENS"  \
    --n_osd_components     "$N_OSD_COMPONENTS"   \
    --n_clean_components   "$N_CLEAN_COMPONENTS" \
    --tasks                "$TASKS"              \
    $( [[ "$RESUME" == "1" ]] && echo "--resume" ) \
    --output_dir           "$OUTPUT_DIR"

echo "=============================================="
echo "Done. Results in: $OUTPUT_DIR/"
echo "=============================================="
