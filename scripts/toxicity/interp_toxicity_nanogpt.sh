#!/bin/bash
#SBATCH --job-name=interp_tox
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/interp_tox-%j.out
#SBATCH --error=slurm/slurm_errors/interp_tox-%j.err

# Interpretability analysis for topo-nanoGPT 125M:
#   1. Logit lens on OSD toxic directions and pruned neurons
#   2. Max-activating examples for top directions and neurons
#   3. Cross-layer OSD subspace similarity
#   4. OSD–pruning alignment measurement
#   5. Tau comparison dashboard
#
# Configurable knobs (override via: sbatch --export=ALL,VAR=value ...):
#
#   TAUS                    comma-separated tau values  (default "0.0,0.5,1.0,3.0,50.0")
#   N_PROMPTS               toxic prompts to load (default 200)
#   N_SELECTIVITY_TOKENS    token budget for activation collection (default 4096)
#   N_OSD_COMPONENTS        toxic PCA components per layer (default 32)
#   N_CLEAN_COMPONENTS      clean PCA components for OSD (default 32)
#   ALIGNMENT_FRAC          pruning fraction for alignment analysis (default 0.2)
#   DATASET                 rtp | toxigen | both (default rtp)
#   NO_LOGIT_LENS           set to 1 to skip logit lens
#   NO_MAX_ACT              set to 1 to skip max-activating examples
#   NO_CROSS_LAYER          set to 1 to skip cross-layer similarity
#   NO_ALIGNMENT            set to 1 to skip OSD–pruning alignment
#   OUTPUT_DIR              output directory
#
# Examples
# --------
# Full analysis (all taus, default settings):
#   sbatch scripts/toxicity/interp_toxicity_nanogpt.sh
#
# Quick single-tau test:
#   sbatch --export=ALL,TAUS="0.0",N_PROMPTS=50 \
#          scripts/toxicity/interp_toxicity_nanogpt.sh
#
# Logit lens only:
#   sbatch --export=ALL,NO_MAX_ACT=1,NO_CROSS_LAYER=1,NO_ALIGNMENT=1 \
#          scripts/toxicity/interp_toxicity_nanogpt.sh

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
N_PROMPTS="${N_PROMPTS:-200}"
N_SELECTIVITY_TOKENS="${N_SELECTIVITY_TOKENS:-4096}"
N_OSD_COMPONENTS="${N_OSD_COMPONENTS:-32}"
N_CLEAN_COMPONENTS="${N_CLEAN_COMPONENTS:-32}"
ALIGNMENT_FRAC="${ALIGNMENT_FRAC:-0.2}"
DATASET="${DATASET:-rtp}"
NO_LOGIT_LENS="${NO_LOGIT_LENS:-0}"
NO_MAX_ACT="${NO_MAX_ACT:-0}"
NO_CROSS_LAYER="${NO_CROSS_LAYER:-0}"
NO_ALIGNMENT="${NO_ALIGNMENT:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/interp_toxicity_nanogpt}"

# -- Job info ------------------------------------------------------------------
echo "=============================================="
echo "Topo NanoGPT — Interpretability Analysis"
echo "=============================================="
echo "Job ID              : ${SLURM_JOB_ID:-local}"
echo "Node                : ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs                : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "taus                : $TAUS"
echo "n_prompts           : $N_PROMPTS"
echo "n_selectivity_tokens: $N_SELECTIVITY_TOKENS"
echo "n_osd_components    : $N_OSD_COMPONENTS"
echo "n_clean_components  : $N_CLEAN_COMPONENTS"
echo "alignment_frac      : $ALIGNMENT_FRAC"
echo "dataset             : $DATASET"
echo "no_logit_lens       : $NO_LOGIT_LENS"
echo "no_max_act          : $NO_MAX_ACT"
echo "no_cross_layer      : $NO_CROSS_LAYER"
echo "no_alignment        : $NO_ALIGNMENT"
echo "output_dir          : $OUTPUT_DIR"
echo "=============================================="

cd "$EXPERIMENT_DIR"
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p slurm/slurm_outputs
mkdir -p slurm/slurm_errors

srun python -u src/toxicity/interp_toxicity_nanogpt.py \
    --taus                   "$TAUS"                    \
    --n_prompts              "$N_PROMPTS"               \
    --n_selectivity_tokens   "$N_SELECTIVITY_TOKENS"    \
    --n_osd_components       "$N_OSD_COMPONENTS"        \
    --n_clean_components     "$N_CLEAN_COMPONENTS"      \
    --alignment_frac         "$ALIGNMENT_FRAC"          \
    --dataset                "$DATASET"                 \
    $( [[ "$NO_LOGIT_LENS"  == "1" ]] && echo "--no_logit_lens"  ) \
    $( [[ "$NO_MAX_ACT"     == "1" ]] && echo "--no_max_act"     ) \
    $( [[ "$NO_CROSS_LAYER" == "1" ]] && echo "--no_cross_layer" ) \
    $( [[ "$NO_ALIGNMENT"   == "1" ]] && echo "--no_alignment"   ) \
    --output_dir             "$OUTPUT_DIR"

echo "=============================================="
echo "Done. Results in: $OUTPUT_DIR/"
echo "=============================================="
