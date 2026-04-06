#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# replot_toxicity_nanogpt.sh
# Re-generate all toxicity benchmark plots from previously saved JSON results.
# No GPU required — pure matplotlib / numpy.
#
# Usage (single directory):
#   sbatch scripts/misc/replot_toxicity_nanogpt.sh
#
# Replot EVERY output directory at once (original + all quantizations):
#   sbatch --export=ALL,ALL_DIRS=1 scripts/misc/replot_toxicity_nanogpt.sh
#
# Override output directory:
#   sbatch --export=ALL,OUTPUT_DIR=/path/to/dir scripts/misc/replot_toxicity_nanogpt.sh
#
# Skip selectivity sub-plots:
#   sbatch --export=ALL,NO_SELECTIVITY=1 scripts/misc/replot_toxicity_nanogpt.sh
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=replot_tox
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/replot_toxicity-%j.out
#SBATCH --error=slurm/slurm_errors/replot_toxicity-%j.err
#SBATCH --partition=overcap
#SBATCH --account=overcap
#SBATCH --qos=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00

set -euo pipefail

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"
cd "$EXPERIMENT_DIR"

# ── Python env ────────────────────────────────────────────────────────────────
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm

# ── Configurable variables ────────────────────────────────────────────────────
ALL_DIRS="${ALL_DIRS:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_nanogpt}"
NO_SELECTIVITY="${NO_SELECTIVITY:-0}"

mkdir -p slurm/slurm_outputs slurm/slurm_errors

# ── Build the list of directories to replot ───────────────────────────────────
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

# ── Helper: replot one directory ─────────────────────────────────────────────
replot_one() {
    local dir="$1"
    if [[ ! -d "$dir" ]]; then
        echo "  [skip] directory not found: $dir"
        return
    fi
    if [[ ! -f "$dir/results.json" ]]; then
        echo "  [skip] results.json not found in $dir"
        return
    fi
    local args=(--output_dir "$dir")
    [[ "$NO_SELECTIVITY" == "1" ]] && args+=(--no_selectivity)
    echo "========================================"
    echo "Replotting: $dir"
    echo "Selectivity: $([[ "$NO_SELECTIVITY" == "1" ]] && echo disabled || echo enabled)"
    echo "========================================"
    python src/toxicity/replot_toxicity_nanogpt.py "${args[@]}"
    echo "Done → $dir"
    echo
}

# ── Run ───────────────────────────────────────────────────────────────────────
for d in "${DIRS[@]}"; do
    replot_one "$d"
done

echo "All replot runs complete."
