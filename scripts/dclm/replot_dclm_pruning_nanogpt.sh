#!/bin/bash
#SBATCH --job-name=replot_dclm_pruning
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=tail-lab
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --qos=short
#SBATCH --output=slurm/slurm_outputs/replot_dclm_pruning-%j.out
#SBATCH --error=slurm/slurm_errors/replot_dclm_pruning-%j.err

# Re-generate all DCLM cross-task pruning plots from saved JSON results.
# No GPU required — reads JSON files and produces matplotlib figures.
#
# Usage:
#   sbatch scripts/dclm/replot_dclm_pruning_nanogpt.sh
#   sbatch --export=ALL,OUTPUT_DIR=path/to/results scripts/dclm/replot_dclm_pruning_nanogpt.sh

set -euo pipefail

# -- Environment ---------------------------------------------------------------
source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
export PYTHONIOENCODING=utf-8

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/dclm_pruning_nanogpt}"

# -- Job info ------------------------------------------------------------------
echo "=============================================="
echo "Topo NanoGPT -- DCLM Pruning Replot"
echo "=============================================="
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURM_NODELIST:-$(hostname)}"
echo "output_dir : $OUTPUT_DIR"
echo "=============================================="

cd "$EXPERIMENT_DIR"
mkdir -p slurm/slurm_outputs
mkdir -p slurm/slurm_errors

srun python -u src/dclm/replot_dclm_pruning_nanogpt.py \
    --output_dir "$OUTPUT_DIR"

echo "=============================================="
echo "Done. Plots in: $OUTPUT_DIR/"
echo "=============================================="
