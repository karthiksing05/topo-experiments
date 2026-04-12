#!/bin/bash
#SBATCH --job-name=eval_tox_tech_450m
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
#SBATCH --output=slurm/slurm_outputs/eval_tox_tech_450m-%j.out
#SBATCH --error=slurm/slurm_errors/eval_tox_tech_450m-%j.err

# Technique comparison for topo-nanoGPT 450M detoxification:
#   per-layer pruning, global pruning,
#   per-layer DAA, global DAA,
#   per-layer OSD, global OSD,
#   topo-region pruning, topo smoothed DAA, topo spectral cluster,
#   activation steering, topo lowrank SVD, topo freq detox, lowrank toxic projection
# on:
#   RealToxicityPrompts  (allenai/real-toxicity-prompts)
#   ToxiGen              (microsoft/toxigen)
# scored by Detoxify (always) and optionally LlamaGuard.
#
# Configurable knobs (override via: sbatch --export=ALL,VAR=value ...):
#
#   TAUS                  comma-separated tau values  (default "0,30722,307226")
#   STEP                  checkpoint step to load     (default 5960)
#   FRACS                 comma-separated intervention fractions  (default "0.0,0.05,...,0.5")
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
#   NO_TOPO               set to 1 to skip topo-aware methods
#   NO_STEERING           set to 1 to skip activation steering
#   NO_LOWRANK            set to 1 to skip low-rank detox methods
#   NO_LLAMAGUARD         set to 1 to skip LlamaGuard scoring
#   NO_TOXIGEN            set to 1 to skip ToxiGen evaluation
#   NO_RTP                set to 1 to skip RealToxicityPrompts evaluation
#   LLAMAGUARD_MODEL      HF model path/id for LlamaGuard (optional)
#   RESUME                set to 1 to skip methods whose results already exist
#   OUTPUT_DIR            output directory
#
# Examples
# --------
# Quick sanity check (tau=0 only, RTP only, 50 prompts):
#   sbatch --export=ALL,N_PROMPTS=50,TAUS="0",NO_PRUNING=1,NO_TOXIGEN=1 \
#          scripts/toxicity/eval_toxicity_techniques_nanogpt_450m.sh
#
# Only pruning + OSD (skip DAA, topo, steering, lowrank, llamaguard):
#   sbatch --export=ALL,NO_DAA=1,NO_TOPO=1,NO_STEERING=1,NO_LOWRANK=1,NO_LLAMAGUARD=1,RESUME=1 \
#          scripts/toxicity/eval_toxicity_techniques_nanogpt_450m.sh
#
# Resume interrupted run:
#   sbatch --export=ALL,RESUME=1 \
#          scripts/toxicity/eval_toxicity_techniques_nanogpt_450m.sh

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
TAUS="${TAUS:-0,30722,307226}"
STEP="${STEP:-5960}"
FRACS="${FRACS:-0.0,0.05,0.1,0.15,0.2}"
N_PROMPTS="${N_PROMPTS:-300}"
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
NO_TOPO="${NO_TOPO:-0}"
NO_STEERING="${NO_STEERING:-0}"
NO_LOWRANK="${NO_LOWRANK:-0}"
NO_LLAMAGUARD="${NO_LLAMAGUARD:-0}"
NO_TOXIGEN="${NO_TOXIGEN:-0}"
NO_RTP="${NO_RTP:-0}"
LLAMAGUARD_MODEL="${LLAMAGUARD_MODEL:-}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_techniques_nanogpt_450m}"

# -- Job info ------------------------------------------------------------------
echo "=============================================="
echo "Topo NanoGPT 450M -- Technique Comparison Eval"
echo "=============================================="
echo "Job ID              : ${SLURM_JOB_ID:-local}"
echo "Node                : ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "HF cache            : $HF_CACHE_DIR"
echo "taus                : $TAUS"
echo "step                : $STEP"
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
echo "no_topo             : $NO_TOPO"
echo "no_steering         : $NO_STEERING"
echo "no_lowrank          : $NO_LOWRANK"
echo "no_llamaguard       : $NO_LLAMAGUARD"
echo "no_toxigen          : $NO_TOXIGEN"
echo "no_rtp              : $NO_RTP"
echo "llamaguard_model    : ${LLAMAGUARD_MODEL:-(default)}"
echo "resume              : $RESUME"
echo "output_dir          : $OUTPUT_DIR"
echo "=============================================="

cd "$EXPERIMENT_DIR"
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p slurm/slurm_outputs
mkdir -p slurm/slurm_errors

# Pass LlamaGuard model only if set
LLAMAGUARD_ARGS=()
if [[ -n "$LLAMAGUARD_MODEL" ]]; then
    LLAMAGUARD_ARGS+=(--llamaguard_model "$LLAMAGUARD_MODEL")
fi

srun python -u src/toxicity/eval_toxicity_techniques_nanogpt_450m.py \
    --taus                   "$TAUS"                  \
    --step                   "$STEP"                  \
    --fracs                  "$FRACS"                 \
    --n_prompts              "$N_PROMPTS"             \
    --n_gen                  "$N_GEN"                 \
    --max_new_tokens         "$MAX_NEW_TOKENS"        \
    --temperature            "$TEMPERATURE"           \
    --top_k                  "$TOP_K"                 \
    --n_selectivity_tokens   "$N_SELECTIVITY_TOKENS"  \
    --n_osd_components       "$N_OSD_COMPONENTS"      \
    --n_clean_components     "$N_CLEAN_COMPONENTS"    \
    $( [[ "$NO_PRUNING"    == "1" ]] && echo "--no_pruning"    ) \
    $( [[ "$NO_DAA"        == "1" ]] && echo "--no_daa"        ) \
    $( [[ "$NO_OSD"        == "1" ]] && echo "--no_osd"        ) \
    $( [[ "$NO_TOPO"       == "1" ]] && echo "--no_topo"       ) \
    $( [[ "$NO_STEERING"   == "1" ]] && echo "--no_steering"   ) \
    $( [[ "$NO_LOWRANK"    == "1" ]] && echo "--no_lowrank"    ) \
    $( [[ "$NO_LLAMAGUARD" == "1" ]] && echo "--no_llamaguard" ) \
    $( [[ "$NO_TOXIGEN"    == "1" ]] && echo "--no_toxigen"    ) \
    $( [[ "$NO_RTP"        == "1" ]] && echo "--no_rtp"        ) \
    $( [[ "$RESUME"        == "1" ]] && echo "--resume"        ) \
    "${LLAMAGUARD_ARGS[@]}" \
    --output_dir             "$OUTPUT_DIR"

echo "=============================================="
echo "Done. Results in: $OUTPUT_DIR/"
echo "=============================================="
