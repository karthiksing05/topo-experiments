#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# replot_toxicity_techniques_nanogpt.sh
# Re-generate all technique-comparison toxicity plots from saved JSON results.
# No GPU required — pure matplotlib / numpy.
#
# Usage:
#   sbatch scripts/toxicity/replot_toxicity_techniques_nanogpt.sh
#
# Override input directory:
#   sbatch --export=ALL,INPUT_DIR=/path/to/dir \
#          scripts/toxicity/replot_toxicity_techniques_nanogpt.sh
#
# Skip subsets:
#   sbatch --export=ALL,NO_SELECTIVITY=1,NO_CORTICAL=1 \
#          scripts/toxicity/replot_toxicity_techniques_nanogpt.sh
# ────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=replot_tox_tech
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --output=slurm/slurm_outputs/replot_tox_tech-%j.out
#SBATCH --error=slurm/slurm_errors/replot_tox_tech-%j.err
#SBATCH --account=overcap
#SBATCH --partition=overcap
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

# ── Configurable knobs ────────────────────────────────────────────────────────
# NOTE: FRACS contains commas, which conflict with sbatch --export=ALL,VAR=val
# syntax.  Use FRAC_MAX instead (e.g. sbatch --export=ALL,FRAC_MAX=0.2 ...)
# to auto-generate 0.0,0.05,...,FRAC_MAX.  Or set FRACS in your shell first:
#   export FRACS="0.0,0.05,0.1,0.15,0.2" && sbatch --export=ALL ...
INPUT_DIR="${INPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_techniques_nanogpt}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
FRAC_MAX="${FRAC_MAX:-}"
if [[ -n "$FRAC_MAX" ]]; then
    FRACS=$(python3 -c "mx=float('$FRAC_MAX'); print(','.join(str(round(x/100,2)) for x in range(0, int(round(mx*100))+1, 5)))")
else
    FRACS="${FRACS:-0.0,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5}"
fi
NO_BAR="${NO_BAR:-0}"
NO_LINE="${NO_LINE:-0}"
NO_PER_MODEL="${NO_PER_MODEL:-0}"
NO_DUAL_SCORER="${NO_DUAL_SCORER:-0}"
NO_SELECTIVITY="${NO_SELECTIVITY:-0}"
NO_CORTICAL="${NO_CORTICAL:-0}"
NO_DAA_OSD_DEBUG="${NO_DAA_OSD_DEBUG:-0}"
NO_CROSS="${NO_CROSS:-0}"
NO_PER_TECHNIQUE="${NO_PER_TECHNIQUE:-0}"
NO_DAA_OSD_CROSS="${NO_DAA_OSD_CROSS:-0}"
NO_DATASET_COMPARE="${NO_DATASET_COMPARE:-0}"

mkdir -p slurm/slurm_outputs slurm/slurm_errors

# ── Build CLI args ────────────────────────────────────────────────────────────
ARGS=(--input_dir "$INPUT_DIR" --fracs "$FRACS")
[[ -n "$OUTPUT_DIR" ]]            && ARGS+=(--output_dir "$OUTPUT_DIR")
[[ "$NO_BAR"            == "1" ]] && ARGS+=(--no_bar)
[[ "$NO_LINE"           == "1" ]] && ARGS+=(--no_line)
[[ "$NO_PER_MODEL"      == "1" ]] && ARGS+=(--no_per_model)
[[ "$NO_DUAL_SCORER"    == "1" ]] && ARGS+=(--no_dual_scorer)
[[ "$NO_SELECTIVITY"    == "1" ]] && ARGS+=(--no_selectivity)
[[ "$NO_CORTICAL"       == "1" ]] && ARGS+=(--no_cortical)
[[ "$NO_DAA_OSD_DEBUG"  == "1" ]] && ARGS+=(--no_daa_osd_debug)
[[ "$NO_CROSS"          == "1" ]] && ARGS+=(--no_cross)
[[ "$NO_PER_TECHNIQUE"  == "1" ]] && ARGS+=(--no_per_technique)
[[ "$NO_DAA_OSD_CROSS"  == "1" ]] && ARGS+=(--no_daa_osd_cross)
[[ "$NO_DATASET_COMPARE" == "1" ]] && ARGS+=(--no_dataset_compare)

echo "=============================================="
echo "Replot: toxicity techniques (125M)"
echo "=============================================="
echo "input_dir  : $INPUT_DIR"
echo "output_dir : ${OUTPUT_DIR:-(same as input)}"
echo "=============================================="

srun python -u src/toxicity/replot_toxicity_techniques_nanogpt.py "${ARGS[@]}"

echo "=============================================="
echo "Done."
echo "=============================================="
