#!/bin/bash
#SBATCH --job-name=corner_ET_best10
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=02:00:00
#SBATCH --mem=16G

set -eo pipefail
ROOT=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo
RESULT_ROOT="$ROOT/examples/gnpe_model/inference/inference_result"
cd "$RESULT_ROOT/analysis"
module load GCCcore/12.3.0 Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH="$ROOT:$HOME/.local/lib/python3.11/site-packages"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1
OUT="$RESULT_ROOT/figures/ET_best_best10_20261008_fullspin_nlive250"
mkdir -p "$OUT"
# The existing Python script writes a shared summary.txt: preserve any prior file.
SUMMARY="$RESULT_ROOT/figures/summary.txt"
BACKUP=$(mktemp)
HAD_SUMMARY=0
if [[ -f "$SUMMARY" ]]; then cp -p "$SUMMARY" "$BACKUP"; HAD_SUMMARY=1; fi
restore_summary() {
    if [[ "$HAD_SUMMARY" == 1 ]]; then cp -p "$BACKUP" "$SUMMARY"; else rm -f "$SUMMARY"; fi
    rm -f "$BACKUP"
}
trap restore_summary EXIT
for index in {0..9}; do
    printf -v target 'injection_%04d' "$index"
    test -s "$RESULT_ROOT/posteriors/ET_best/best10_20261007/$target.hdf5"
    test -s "$ROOT/bilby_baseline/bilby_injections/ET_best_best10_20261008_fullspin_nlive250/$target/bilby_result.json"
done
for index in {0..9}; do
    printf -v target 'injection_%04d' "$index"
    echo "START $target $(date -Is)"
    python -u plot_posteriors.py "$target" \
        --dingo "$RESULT_ROOT/posteriors/ET_best/best10_20261007/$target.hdf5" \
        --bilby "$ROOT/bilby_baseline/bilby_injections/ET_best_best10_20261008_fullspin_nlive250/$target/bilby_result.json" \
        --outdir "$OUT"
    test -s "$OUT/$target/${target}_ET_best_dingo_vs_bilby.png"
    cp "$SUMMARY" "$OUT/$target/summary.txt"
    echo "DONE $target $(date -Is)"
done
echo "All 10 comparison corners completed."
