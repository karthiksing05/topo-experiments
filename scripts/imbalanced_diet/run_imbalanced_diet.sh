#!/bin/bash
#SBATCH --job-name=imbalanced_diet_topo
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH --time=4:00:00
#SBATCH --qos=short
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm/slurm_outputs/imbalanced_diet_topo-%j.out
#SBATCH --error=slurm/slurm_errors/imbalanced_diet_topo-%j.err

# ── Environment ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"

export TORCH_HOME="${EXPERIMENT_DIR}/.torch_cache"

echo "=========================================================="
echo "  Imbalanced Diet Topographic FashionMNIST Experiment"
echo "  4 classes: T-shirt / Trouser / Sneaker / Bag"
echo "  4 variants: baseline | topo_r2 | topo_r4 | topo_r8  (τ=5)"
echo "  6 diets: balanced | top/bottom/footwear/bag dominant | extreme"
echo "=========================================================="
echo "Job ID : $SLURM_JOB_ID"
echo "Node   : $SLURM_NODELIST"
echo "GPUs   : $CUDA_VISIBLE_DEVICES"
echo "Time   : $(date)"
echo "=========================================================="

cd "$EXPERIMENT_DIR"
mkdir -p slurm/slurm_outputs
mkdir -p slurm/slurm_errors
mkdir -p outputs/imbalanced_diet/checkpoints
mkdir -p outputs/imbalanced_diet/results
mkdir -p outputs/imbalanced_diet/figures

# ── Configurable overrides via sbatch --export= ────────────────────────────────
CONFIG_FILE="${CONFIG_FILE:-${EXPERIMENT_DIR}/configs/imbalanced_diet.json}"
DATA_DIR="${DATA_DIR:-}"
OUTPUT_DIR_OVERRIDE="${OUTPUT_DIR_OVERRIDE:-}"
EPOCHS="${EPOCHS:-}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
LR="${LR:-}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-}"

echo ""
echo "config  : ${CONFIG_FILE}"
echo "device  : ${DEVICE}"
[[ -n "$EPOCHS"        ]] && echo "epochs  : ${EPOCHS} (override)"
[[ -n "$SEED"          ]] && echo "seed    : ${SEED}   (override)"
echo ""

OVERRIDE_ARGS=""
[[ -n "$DATA_DIR"            ]] && OVERRIDE_ARGS+=" --data-dir ${DATA_DIR}"
[[ -n "$OUTPUT_DIR_OVERRIDE" ]] && OVERRIDE_ARGS+=" --output-dir ${OUTPUT_DIR_OVERRIDE}"
[[ -n "$EPOCHS"              ]] && OVERRIDE_ARGS+=" --epochs ${EPOCHS}"
[[ -n "$TOTAL_SAMPLES"       ]] && OVERRIDE_ARGS+=" --total-samples ${TOTAL_SAMPLES}"
[[ -n "$BATCH_SIZE"          ]] && OVERRIDE_ARGS+=" --batch-size ${BATCH_SIZE}"
[[ -n "$LR"                  ]] && OVERRIDE_ARGS+=" --lr ${LR}"
[[ -n "$DEVICE"              ]] && OVERRIDE_ARGS+=" --device ${DEVICE}"
[[ -n "$SEED"                ]] && OVERRIDE_ARGS+=" --seed ${SEED}"

# ── Phase 1: Training (24 combinations = 4 variants × 6 diets) ────────────────
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1: Training"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

srun python -u src/imbalanced_diet/imbalanced_diet_topo.py \
    --config "${CONFIG_FILE}" \
    ${OVERRIDE_ARGS}

TRAIN_EXIT=$?
echo ""
if [[ $TRAIN_EXIT -ne 0 ]]; then
    echo "ERROR: Training script exited with code ${TRAIN_EXIT}. Skipping analysis."
    exit $TRAIN_EXIT
fi

echo "Training complete at $(date)"

# ── Phase 2: Analysis & Figures ───────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2: Analysis & Figure Generation"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

RESULTS_FILE="${EXPERIMENT_DIR}/outputs/imbalanced_diet/results/imbalanced_diet_results_latest.json"
FIGURES_DIR="${EXPERIMENT_DIR}/outputs/imbalanced_diet/figures"

srun python -u src/imbalanced_diet/analyze_imbalanced_diet.py \
    --results  "${RESULTS_FILE}" \
    --output-dir "${FIGURES_DIR}" \
    --hidden-size 256

ANALYZE_EXIT=$?

echo ""
echo "=========================================================="
echo "  Experiment complete at $(date)"
echo ""
echo "  Results : outputs/imbalanced_diet/results/"
echo "  Figures : outputs/imbalanced_diet/figures/"
echo "  Logs    : slurm/slurm_outputs/imbalanced_diet_topo-${SLURM_JOB_ID}.out"
echo "=========================================================="

exit $ANALYZE_EXIT
