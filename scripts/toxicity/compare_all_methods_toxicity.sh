#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# compare_all_methods_toxicity.sh
# Generate per-model all-methods comparison plots (toxicity / PPL / val-loss)
# from previously saved JSON results.  No GPU required.
#
# Usage:
#   sbatch scripts/toxicity/compare_all_methods_toxicity.sh
#
# Override output directory:
#   sbatch --export=ALL,OUTPUT_DIR=/path/to/dir scripts/toxicity/compare_all_methods_toxicity.sh
#
# Override plot subdirectory (default: all_methods_comparison):
#   sbatch --export=ALL,OUT_SUBDIR=my_subdir scripts/toxicity/compare_all_methods_toxicity.sh
#
# Run across all quantization output dirs at once:
#   sbatch --export=ALL,ALL_DIRS=1 scripts/toxicity/compare_all_methods_toxicity.sh
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=cmp_methods_tox
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/compare_all_methods_toxicity-%j.out
#SBATCH --error=slurm/slurm_errors/compare_all_methods_toxicity-%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00

set -euo pipefail

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"
cd "$EXPERIMENT_DIR"

# ── Python env ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Configurable variables ────────────────────────────────────────────────────
ALL_DIRS="${ALL_DIRS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_nanogpt}"
OUT_SUBDIR="${OUT_SUBDIR:-all_methods_comparison}"

mkdir -p slurm/slurm_outputs slurm/slurm_errors

# ── Build list of directories to process ─────────────────────────────────────
if [[ "$ALL_DIRS" == "1" ]]; then
    DIRS=(
        "${EXPERIMENT_DIR}/outputs/toxicity_nanogpt"
        "${EXPERIMENT_DIR}/outputs/toxicity_nanogpt_quantized_fp16"
        "${EXPERIMENT_DIR}/outputs/toxicity_nanogpt_quantized_bf16"
        "${EXPERIMENT_DIR}/outputs/toxicity_nanogpt_quantized_int8"
        "${EXPERIMENT_DIR}/outputs/toxicity_nanogpt_quantized_int4"
    )
else
    DIRS=("$OUTPUT_DIR")
fi

# ── Job info ──────────────────────────────────────────────────────────────────
echo "=========================================="
echo "Topo NanoGPT -- All-Methods Toxicity Comparison"
echo "=========================================="
echo "Directories : ${DIRS[*]}"
echo "Out subdir  : $OUT_SUBDIR"
echo "=========================================="
echo ""

# ── Run for each directory ────────────────────────────────────────────────────
for DIR in "${DIRS[@]}"; do
    if [[ ! -d "$DIR" ]]; then
        echo "Directory not found, skipping: $DIR"
        continue
    fi
    echo "--- Processing: $DIR ---"
    python3 src/toxicity/compare_all_methods_toxicity.py \
        --output_dir "$DIR" \
        --out_subdir "$OUT_SUBDIR"
    echo ""
done

echo "=========================================="
echo "Done."
echo "=========================================="
