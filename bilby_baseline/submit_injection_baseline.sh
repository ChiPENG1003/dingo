#!/bin/bash
#SBATCH --job-name=bilby_inj
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=48:00:00
#SBATCH --mem=16G

# Independent bilby/dynesty baseline for ONE Dingo injection.
#
# The injection HDF5 already carries a Dingo posterior; run_baseline.py ignores
# it and reads only context/waveform (the strain d), context/asds (the PSD) and
# the waveform settings, so the same file serves as both the Dingo answer and
# the bilby input.
#
# All 15 parameters are sampled. That is deliberate: a 7-parameter Dingo model
# marginalises over spins/phase/theta_jn (they were drawn from the prior during
# training), so the matching bilby posterior is the 15-D one restricted to those
# 7 marginals -- NOT run_population.py, which fixes the rest to the injected
# truth and therefore produces a conditional posterior instead.
#
# Usage:
#   sbatch submit_injection_baseline.sh
#   RESULT=<path/to/injection.hdf5> OUTDIR=bilby_injections/foo \
#       sbatch submit_injection_baseline.sh

set -eo pipefail

start_time=$(date +%s)
mkdir -p logs
echo "HOSTNAME=$(hostname)  $start_time"

BASE_DIR="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline"
source "$BASE_DIR/env.sh"
cd "$BASE_DIR"

# run_baseline.py's own default --result points at the pre-reorganisation
# inference_result/, which no longer exists, so always pass it explicitly.
POSTERIORS="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors"
RESULT="${RESULT:-$POSTERIORS/1M_7params_O2/single/seed0000_dprior.hdf5}"
OUTDIR="${OUTDIR:-bilby_injections/$(basename "${RESULT%.hdf5}")}"

if [ ! -f "$RESULT" ]; then
    echo "ERROR: no such injection: $RESULT" >&2
    exit 1
fi

# Fail fast and loud on a bad environment rather than 48 h later.
python - "$RESULT" <<'PY'
import sys
import bilby, sklearn  # noqa: F401  -- the Anaconda/pyarrow ABI trap
from dingo.gw.result import Result
assert "anaconda" not in sys.executable.lower(), sys.executable
r = Result(file_name=sys.argv[1])
assert "waveform" in r.context and "asds" in r.context, "injection has no context data"
print(f"preflight OK: {r.interferometers}, f_min={r.domain.f_min}, f_max={r.domain.f_max}")
PY

echo "result -> $RESULT"
echo "outdir -> $OUTDIR"

# --label bilby (not the run_baseline.py default) so the downstream comparison
# scripts, which look for bilby_result.{json,hdf5}, find it without a --bilby flag.
python -u run_baseline.py \
    --result "$RESULT" \
    --outdir "$OUTDIR" \
    --label bilby \
    --nlive "${NLIVE:-1000}" \
    --naccept "${NACCEPT:-60}" \
    --dlogz "${DLOGZ:-0.5}" \
    --npool "${SLURM_CPUS_PER_TASK:-8}"

end_time=$(date +%s)
time_taken=$((end_time-start_time))
echo "Done. $end_time"

printf "Elapsed time: %02d:%02d:%02d\n" \
    $((time_taken / 3600)) \
    $(((time_taken % 3600) / 60)) \
    $((time_taken % 60))
