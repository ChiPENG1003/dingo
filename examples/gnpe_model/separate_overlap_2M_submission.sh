#!/bin/bash
#SBATCH --job-name=separate_overlap_ET_2M
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-gpu
#SBATCH --time=8:00:00
#SBATCH --mem=64G

# Build training_data/waveform_dataset_ET_joint_separate_2M.hdf5: the two signals kept
# separate, to be summed after projection at training time.
#
# Almost pure I/O (read 2 x 32 GB, write 64 GB). The change of basis for signal 2 is a
# single (200 x 200) matmul per chunk, so no waveform is ever decompressed. No --gres,
# so this takes CPUs on a GPU node without holding a card; private-dpnc-cpu is
# chronically saturated (see overlap_combine_submission.sh).

set -eo pipefail

cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
mkdir -p logs/dingo

SIGNAL1=${SIGNAL1:-training_data/waveform_dataset_ET_5M.hdf5}
SIGNAL2=${SIGNAL2:-training_data/waveform_dataset_ET_5M_2.hdf5}
OUT=${OUT:-training_data/waveform_dataset_ET_joint_separate_2M.hdf5}

echo "HOSTNAME=$(hostname)"; echo "PWD=$(pwd)"
echo "SIGNAL1=$SIGNAL1"; echo "SIGNAL2=$SIGNAL2"; echo "OUT=$OUT"
df -h /srv/beegfs/dpnc | tail -1

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}

module purge
module load GCCcore/12.3.0
module load Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages

for f in "$SIGNAL1" "$SIGNAL2"; do
    [ -f "$f" ] || { echo "ERROR: $f does not exist." >&2; exit 1; }
done

echo; echo "=== 1/2 building $OUT ==="
python make_separate_overlap_dataset.py \
    --signal1 "$SIGNAL1" --signal2 "$SIGNAL2" --out "$OUT" \
    --num-samples 2000000

echo; echo "=== 2/2 verifying $OUT ==="
# leave_waveforms_on_disk=True is essential here: the four coefficient arrays are
# 4 x 5e6 x 200 complex, so loading them into RAM needs ~64 GB (and transiently twice
# that while converting to single precision). Training uses the same setting.
# build_dataset (rather than WaveformDataset directly) is what adds derived
# parameters such as chi_eff, so the check has to go through it.
python -c "
import numpy as np, yaml, h5py, resource
from dingo.gw.training.train_builders import build_dataset, set_train_transforms
with h5py.File('$OUT', 'r') as f:
    keys = sorted(f['polarizations'].keys())
    assert len(f['parameters']) == 2000000
print('polarization keys:', keys)
assert keys == ['h_cross','h_cross_2','h_plus','h_plus_2'], keys
settings = yaml.safe_load(open('configs/train_settings_ET_JPE.yaml'))
ds = settings['data']
ds['waveform_dataset_path'] = '$OUT'
wfd = build_dataset(ds, leave_waveforms_on_disk=True)
set_train_transforms(wfd, ds, settings['training']['stage_0']['asd_dataset_path'])
out = wfd[0]
print('inference params:', len(ds['inference_parameters']), '| context:', ds['context_parameters'])
print('batch shapes:', [np.shape(x) for x in out])
assert all(np.isfinite(np.asarray(x)).all() for x in out)
print('peak RSS: %.1f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6))
print('OK')
"
ls -lh training_data/*.hdf5
echo "Done."
