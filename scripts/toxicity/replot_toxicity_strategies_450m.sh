#!/bin/bash
#SBATCH --job-name=replot_tox_strats_450m
#SBATCH --exclude=spot,heistotron,clippy,hal,asimo,kipp,smith,t1000,bb8,jarvis,gideon,ripl-s1,ash,c3po,calculon,eva,johnny5,neo,tars,vicki,ava,jill,walle
#SBATCH --account=overcap
#SBATCH --partition=overcap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --qos=short
#SBATCH --output=slurm/slurm_outputs/replot_tox_strats_450m-%j.out
#SBATCH --error=slurm/slurm_errors/replot_tox_strats_450m-%j.err

# Replot: merge techniques + strategies results for nanoGPT 450M.

set -euo pipefail

source ~/flash/miniconda3/etc/profile.d/conda.sh
conda activate topovlm
export PYTHONIOENCODING=utf-8

EXPERIMENT_DIR="/nethome/ksingara3/flash/topo-experiments"
cd "${EXPERIMENT_DIR}"

FRAC_MAX="${FRAC_MAX:-0.5}"
FRACS="${FRACS:-$(python3 -c "
fmax=float('${FRAC_MAX}')
pts=[round(i*0.05,2) for i in range(int(fmax/0.05)+1)]
if pts[-1]<fmax: pts.append(fmax)
print(','.join(str(round(x,2)) for x in pts))
")}"

TARGET_FRACS="${TARGET_FRACS:-0.1,0.2,0.3}"
TECHNIQUES_DIR="${TECHNIQUES_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_techniques_nanogpt_450m}"
STRATEGIES_DIR="${STRATEGIES_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_strategies_nanogpt_450m}"
OUTPUT_DIR="${OUTPUT_DIR:-${EXPERIMENT_DIR}/outputs/toxicity_comparison_nanogpt_450m}"

echo "=============================================="
echo "Replot: Strategies vs Techniques (450M)"
echo "=============================================="
echo "Techniques dir : ${TECHNIQUES_DIR}"
echo "Strategies dir : ${STRATEGIES_DIR}"
echo "Output dir     : ${OUTPUT_DIR}"
echo "Fracs          : ${FRACS}"
echo "Target fracs   : ${TARGET_FRACS}"
echo "=============================================="

mkdir -p slurm/slurm_outputs slurm/slurm_errors

exec python -u -m src.toxicity.replot_toxicity_strategies \
    --variant 450m \
    --techniques_dir "$TECHNIQUES_DIR" \
    --strategies_dir "$STRATEGIES_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --fracs "$FRACS" \
    --target_fracs "$TARGET_FRACS"
