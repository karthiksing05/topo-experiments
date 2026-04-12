#!/bin/bash
#SBATCH --job-name=eval_tox_strats_450m
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/eval_tox_strats_450m-%j.out
#SBATCH --error=slurm/slurm_errors/eval_tox_strats_450m-%j.err

# Evaluate NEW strategies on topo-nanoGPT 450M checkpoints.

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

# Synchronous CUDA error reporting for debugging
export CUDA_LAUNCH_BLOCKING=1

# -- Configurable knobs --------------------------------------------------------
TAUS="${TAUS:-0,30722,307226}"
STEP="${STEP:-5960}"
N_PROMPTS="${N_PROMPTS:-300}"
N_GEN="${N_GEN:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-200}"
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_K="${TOP_K:-50}"
N_SELECTIVITY_TOKENS="${N_SELECTIVITY_TOKENS:-4096}"

NO_EIGENSHIFT="${NO_EIGENSHIFT:-0}"
NO_SELF_DEBIASING="${NO_SELF_DEBIASING:-0}"
NO_CHARS="${NO_CHARS:-0}"
NO_VOCAB_SHIFTING="${NO_VOCAB_SHIFTING:-0}"
NO_PCT="${NO_PCT:-0}"
NO_LLAMAGUARD="${NO_LLAMAGUARD:-0}"
NO_TOXIGEN="${NO_TOXIGEN:-0}"
NO_RTP="${NO_RTP:-0}"
RESUME="${RESUME:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_strategies_nanogpt_450m}"

FRAC_MAX="${FRAC_MAX:-0.5}"
FRACS="${FRACS:-$(python3 -c "
fmax=float('${FRAC_MAX}')
pts=[round(i*0.05,2) for i in range(int(fmax/0.05)+1)]
if pts[-1]<fmax: pts.append(fmax)
print(','.join(str(round(x,2)) for x in pts))
")}"

# -- Job info ------------------------------------------------------------------
echo "=============================================="
echo "Topo NanoGPT 450M -- New Strategies Eval"
echo "=============================================="
echo "Job ID              : ${SLURM_JOB_ID:-local}"
echo "Date                : $(date)"
echo "Node                : $(hostname)"
echo "GPU                 : $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Taus                : ${TAUS}"
echo "Step                : ${STEP}"
echo "Fracs               : ${FRACS}"
echo "Prompts/dataset     : ${N_PROMPTS}"
echo "Output dir          : ${OUTPUT_DIR}"
echo "=============================================="

cd "${EXPERIMENT_DIR}"
mkdir -p slurm/slurm_outputs slurm/slurm_errors

# -- Build cmd -----------------------------------------------------------------
CMD=(
    python -u -m src.toxicity.eval_toxicity_strategies_nanogpt_450m
    --taus "$TAUS"
    --step "$STEP"
    --fracs "$FRACS"
    --n_prompts "$N_PROMPTS"
    --n_gen "$N_GEN"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --temperature "$TEMPERATURE"
    --top_k "$TOP_K"
    --n_selectivity_tokens "$N_SELECTIVITY_TOKENS"
    --output_dir "$OUTPUT_DIR"
)

[[ "$NO_EIGENSHIFT"     == "1" ]] && CMD+=(--no_eigenshift)
[[ "$NO_SELF_DEBIASING" == "1" ]] && CMD+=(--no_self_debiasing)
[[ "$NO_CHARS"          == "1" ]] && CMD+=(--no_chars)
[[ "$NO_VOCAB_SHIFTING" == "1" ]] && CMD+=(--no_vocab_shifting)
[[ "$NO_PCT"            == "1" ]] && CMD+=(--no_pct)
[[ "$NO_LLAMAGUARD"     == "1" ]] && CMD+=(--no_llamaguard)
[[ "$NO_TOXIGEN"        == "1" ]] && CMD+=(--no_toxigen)
[[ "$NO_RTP"            == "1" ]] && CMD+=(--no_rtp)
[[ "$RESUME"            == "1" ]] && CMD+=(--resume)

echo ""
echo "Command:"
echo "  ${CMD[*]}"
echo ""

exec "${CMD[@]}"
