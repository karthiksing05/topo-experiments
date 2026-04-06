#!/bin/bash
#SBATCH --job-name=eval_toxicity_nanogpt
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
#SBATCH --output=slurm/slurm_outputs/eval_toxicity_nanogpt-%j.out
#SBATCH --error=slurm/slurm_errors/eval_toxicity_nanogpt-%j.err

# Evaluate the topo toxicity benchmark on regular nanoGPT (tau=0.0) and
# topo-regularised nanoGPT checkpoints (tau=0.5, 1.0, 3.0, 50.0).
#
# All checkpoints are downloaded from murtylab/topo-nanogpt-fineweb10B.
# Toxicity is scored with detoxify; outputs are saved to:
#   outputs/toxicity_nanogpt/
#
# Configurable knobs (override via: sbatch --export=VAR=value ...):
#   N_PROMPTS             number of toxic prompts from RealToxicityPrompts (default 200)
#   N_GEN                 number of completions per prompt (default 1)
#   MAX_NEW_TOKENS        new tokens to generate per completion (default 200)
#   TEMPERATURE           sampling temperature (default 1.0)
#   TOP_K                 top-k sampling cutoff; 0 = greedy (default 50)
#   TAUS                  comma-separated tau values (default "0.0,0.5,1.0,3.0,50.0")
#   PRUNING_FRACS         comma-separated pruning fractions (default "0.05,0.1,0.15,0.2")
#   N_SELECTIVITY_TOKENS  token budget for activation collection (default 4096)
#   NO_PRUNING            set to 1 to skip the pruning sweep (faster)
#   NO_GLOBAL_PRUNING     set to 1 to skip the cross-layer global pruning sweep
#   SVD_PRUNING_FRACS     comma-separated SVD pruning fractions (default "0.05,0.1,0.15,0.2")
#   NO_SVD_PRUNING        set to 1 to skip SVD pruning sweep
#   AMP_FACTOR            amplification factor for toxic neurons (default 5.0)
#   AMP_FRACS             comma-separated fractions for amplification sweep (default "0.05,0.1,0.15,0.2")
#   NO_AMPLIFICATION      set to 1 to skip the amplification sweep
#   ATT_FACTOR            attenuation factor for toxic neurons, must be <1 (default 0.2)
#   ATT_FRACS             comma-separated fractions for attenuation sweep (default "0.05,0.1,0.15,0.2")
#   NO_ATTENUATION        set to 1 to skip the attenuation sweep
#   PCA_PRUNING_FRACS     comma-separated PC fractions to remove (default "0.05,0.1,0.15,0.2")
#   N_PCA_COMPONENTS      number of PCA components per layer (default 32)
#   NO_PCA_PRUNING        set to 1 to skip the PCA pruning sweep
#   DAA_PRUNING_FRACS     comma-separated projection-strength values for DAA sweep (default "0.05,0.1,0.15,0.2")
#   NO_DAA_PRUNING        set to 1 to skip the DAA pruning sweep
#   OSD_PRUNING_FRACS     comma-separated PC fractions for OSD sweep (default "0.05,0.1,0.15,0.2")
#   N_OSD_COMPONENTS      number of OSD toxic components per layer (default 32)
#   N_CLEAN_COMPONENTS    number of clean PCA components for OSD subspace (default 32)
#   NO_OSD_PRUNING        set to 1 to skip the OSD pruning sweep
#   REPENG_ALPHAS         comma-separated steering coefficients α for rep-eng (default "1,2,5,10,20")
#   NO_REPENG             set to 1 to skip the representation-engineering steering sweep
#   RESUME                set to 1 to skip per-tau sweeps whose JSON outputs already exist
#   OUTPUT_DIR            output directory (default: outputs/toxicity_nanogpt)
#
# Example — run with more prompts on a specific tau subset, no pruning:
#   sbatch --export=N_PROMPTS=500,TAUS="0.0,1.0,50.0",NO_PRUNING=1 \
#          scripts/toxicity/eval_toxicity_nanogpt.sh
#
# Example — resume an interrupted run (skip completed taus):
#   sbatch --export=ALL,RESUME=1 scripts/toxicity/eval_toxicity_nanogpt.sh

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
N_PROMPTS="${N_PROMPTS:-200}"
N_GEN="${N_GEN:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
TAUS="${TAUS:-0.0,0.5,1.0,3.0,50.0}"
PRUNING_FRACS="${PRUNING_FRACS:-0.05,0.1,0.15,0.2}"
N_SELECTIVITY_TOKENS="${N_SELECTIVITY_TOKENS:-4096}"
NO_PRUNING="${NO_PRUNING:-0}"
NO_GLOBAL_PRUNING="${NO_GLOBAL_PRUNING:-0}"
NO_SVD_PRUNING="${NO_SVD_PRUNING:-0}"
AMP_FACTOR="${AMP_FACTOR:-5.0}"
AMP_FRACS="${AMP_FRACS:-0.05,0.1,0.15,0.2}"
NO_AMPLIFICATION="${NO_AMPLIFICATION:-0}"
ATT_FACTOR="${ATT_FACTOR:-0.2}"
ATT_FRACS="${ATT_FRACS:-0.05,0.1,0.15,0.2}"
NO_ATTENUATION="${NO_ATTENUATION:-0}"
PCA_PRUNING_FRACS="${PCA_PRUNING_FRACS:-0.05,0.1,0.15,0.2}"
N_PCA_COMPONENTS="${N_PCA_COMPONENTS:-32}"
NO_PCA_PRUNING="${NO_PCA_PRUNING:-0}"
DAA_PRUNING_FRACS="${DAA_PRUNING_FRACS:-0.05,0.1,0.15,0.2}"
NO_DAA_PRUNING="${NO_DAA_PRUNING:-0}"
OSD_PRUNING_FRACS="${OSD_PRUNING_FRACS:-0.05,0.1,0.15,0.2}"
N_OSD_COMPONENTS="${N_OSD_COMPONENTS:-32}"
N_CLEAN_COMPONENTS="${N_CLEAN_COMPONENTS:-32}"
NO_OSD_PRUNING="${NO_OSD_PRUNING:-0}"
REPENG_ALPHAS="${REPENG_ALPHAS:-1,2,5,10,20}"
NO_REPENG="${NO_REPENG:-0}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_nanogpt}"

# -- Job info ------------------------------------------------------------------
echo "=========================================="
echo "Topo NanoGPT -- Toxicity Benchmark"
echo "=========================================="
echo "Job ID          : ${SLURM_JOB_ID:-local}"
echo "Node            : ${SLURM_NODELIST:-$(hostname)}"
echo "GPUs            : ${CUDA_VISIBLE_DEVICES:-N/A}"
echo "HF cache        : $HF_CACHE_DIR"
echo "n_prompts       : $N_PROMPTS"
echo "n_gen           : $N_GEN"
echo "max_new_tokens  : $MAX_NEW_TOKENS"
echo "temperature     : $TEMPERATURE"
echo "top_k           : $TOP_K"
echo "taus            : $TAUS"
echo "pruning_fracs   : $PRUNING_FRACS"
echo "selectivity_toks: $N_SELECTIVITY_TOKENS"
echo "no_pruning      : $NO_PRUNING"
echo "no_global_pruning: $NO_GLOBAL_PRUNING"
echo "no_svd_pruning  : $NO_SVD_PRUNING"
echo "amp_factor      : $AMP_FACTOR"
echo "amp_fracs       : $AMP_FRACS"
echo "no_amplification: $NO_AMPLIFICATION"
echo "att_factor      : $ATT_FACTOR"
echo "att_fracs       : $ATT_FRACS"
echo "no_attenuation  : $NO_ATTENUATION"
echo "pca_fracs       : $PCA_PRUNING_FRACS"
echo "n_pca_components: $N_PCA_COMPONENTS"
echo "no_pca_pruning  : $NO_PCA_PRUNING"
echo "daa_prune_fracs : $DAA_PRUNING_FRACS"
echo "no_daa_pruning  : $NO_DAA_PRUNING"
echo "osd_prune_fracs : $OSD_PRUNING_FRACS"
echo "n_osd_components: $N_OSD_COMPONENTS"
echo "n_clean_comps   : $N_CLEAN_COMPONENTS"
echo "no_osd_pruning  : $NO_OSD_PRUNING"
echo "repeng_alphas   : $REPENG_ALPHAS"
echo "no_repeng       : $NO_REPENG"
echo "resume          : $RESUME"
echo "output_dir       : $OUTPUT_DIR"
echo "=========================================="

cd "$EXPERIMENT_DIR"
mkdir -p "${HF_CACHE_DIR}"
mkdir -p "${OUTPUT_DIR}"

srun python -u src/toxicity/eval_toxicity_nanogpt.py \
    --n_prompts      "$N_PROMPTS"      \
    --n_gen          "$N_GEN"          \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature    "$TEMPERATURE"    \
    --top_k          "$TOP_K"          \
    --taus           "$TAUS"           \
    --pruning_fracs  "$PRUNING_FRACS"  \
    --n_selectivity_tokens "$N_SELECTIVITY_TOKENS" \
    $( [[ "$NO_PRUNING"        == "1" ]] && echo "--no_pruning" ) \
    $( [[ "$NO_GLOBAL_PRUNING" == "1" ]] && echo "--no_global_pruning" ) \
    $( [[ "$NO_SVD_PRUNING"    == "1" ]] && echo "--no_svd_pruning" ) \
    --amp_factor     "$AMP_FACTOR" \
    --amp_fracs      "$AMP_FRACS" \
    $( [[ "$NO_AMPLIFICATION"  == "1" ]] && echo "--no_amplification" ) \
    --att_factor     "$ATT_FACTOR" \
    --att_fracs      "$ATT_FRACS" \
    $( [[ "$NO_ATTENUATION"    == "1" ]] && echo "--no_attenuation" ) \
    --pca_pruning_fracs "$PCA_PRUNING_FRACS" \
    --n_pca_components  "$N_PCA_COMPONENTS" \
    $( [[ "$NO_PCA_PRUNING"    == "1" ]] && echo "--no_pca_pruning" ) \
    --daa_pruning_fracs "$DAA_PRUNING_FRACS" \
    $( [[ "$NO_DAA_PRUNING"    == "1" ]] && echo "--no_daa_pruning" ) \
    --osd_pruning_fracs "$OSD_PRUNING_FRACS" \
    --n_osd_components  "$N_OSD_COMPONENTS" \
    --n_clean_components "$N_CLEAN_COMPONENTS" \
    $( [[ "$NO_OSD_PRUNING"    == "1" ]] && echo "--no_osd_pruning" ) \
    --repeng_alphas     "$REPENG_ALPHAS" \
    $( [[ "$NO_REPENG"         == "1" ]] && echo "--no_repeng" ) \
    $( [[ "$RESUME"            == "1" ]] && echo "--resume" ) \
    --output_dir     "$OUTPUT_DIR"

echo "=========================================="
echo "Done. Results in: $OUTPUT_DIR/"
echo "=========================================="
