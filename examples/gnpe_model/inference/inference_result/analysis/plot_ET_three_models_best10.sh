#!/bin/bash
#SBATCH --job-name=corner_ET_three_best10
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
python -u - <<'CHECK'
import os
import h5py
import numpy as np
root = os.path.abspath('..')
for i in range(10):
    name = f'injection_{i:04d}.hdf5'
    a = os.path.join(root, 'posteriors/ET_best/best10_20261007', name)
    b = os.path.join(root, 'posteriors/ET_5M_complete_best140/best10_20261021', name)
    with h5py.File(a) as old, h5py.File(b) as new:
        print(name, 'old model:', old.attrs.get('model_main'), 'new model:', new.attrs.get('model_main'), flush=True)
        assert str(old.attrs['model_main']).endswith('main_train_dir_ET/best_model.pt')
        assert str(new.attrs['model_main']).endswith('main_train_dir_ET_5M_complete/best_model.pt')
        assert int(old['injection_parameters'].attrs['injection_seed']) == 10000+i
        for group in ('injection_parameters', 'context/waveform', 'context/asds'):
            for key in old[group]:
                assert np.array_equal(old[group][key][()], new[group][key][()]), (name,group,key)
print('All ten original truths, strain arrays, and ASDs match exactly.', flush=True)
CHECK
OUT="$RESULT_ROOT/figures/ET_three_models_best10_20261021"
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
        --dingo "$RESULT_ROOT/posteriors/ET_5M_complete_best140/best10_20261021/$target.hdf5" "$RESULT_ROOT/posteriors/ET_best/best10_20261007/$target.hdf5" \
        --bilby "$ROOT/bilby_baseline/bilby_injections/ET_best_best10_20261008_fullspin_nlive250/$target/bilby_result.json" \
        --outdir "$OUT"
    test -n "$(find "$OUT/$target" -name "*_dingo_vs_bilby.png" -size +0c -print -quit)"
    cp "$SUMMARY" "$OUT/$target/summary.txt"
    echo "DONE $target $(date -Is)"
done
echo "All 10 comparison corners completed."
