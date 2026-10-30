#!/bin/bash
#SBATCH --job-name=bilby_ET_chi
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=48:00:00
#SBATCH --mem=16G

# Bilby/dynesty full-spin baseline for one ET Dingo injection.
# Bilby samples the full component-spin parameterization.
# chi_eff is derived afterwards from the joint posterior samples.
#
# Usage:
#   sbatch submit_ET_baseline.sh
# or
#   RESULT=<path/to/injection.hdf5> \
#   NLIVE=1000 \
#   sbatch submit_ET_baseline.sh

set -eo pipefail

start_time=$(date +%s)

mkdir -p logs

echo "HOSTNAME=$(hostname)"
echo "START=$(date)"

BASE_DIR="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline"

# One tab-separated record per invocation; flock serializes concurrent writers.
SUMMARY_FILE="${SUMMARY_FILE:-$BASE_DIR/logs/bilby_task_summary.tsv}"
record_task_summary() {
    local code=$?
    trap - EXIT
    set +e
    local end_time elapsed duration status sampled task
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    printf -v duration '%02d:%02d:%02d' "$((elapsed / 3600))" "$(((elapsed % 3600) / 60))" "$((elapsed % 60))"
    status=FAILED
    [[ "$code" == 0 ]] && status=COMPLETED
    sampled=unavailable
    if [[ -n "${RUN_LOG:-}" && -f "$RUN_LOG" ]]; then
        sampled=$(sed -n 's/^sampling [0-9][0-9]* parameters: //p' "$RUN_LOG" | tail -1)
        sampled="${sampled:-unavailable}"
    fi
    task="${SLURM_JOB_NAME:-bilby}_$(basename "${RESULT:-unknown}" .hdf5)"
    mkdir -p "$(dirname "$SUMMARY_FILE")"
    (
        flock -x 9 || exit 1
        if [[ ! -s "$SUMMARY_FILE" ]]; then
            printf 'job_id\ttask_name\tsampled_parameters\ttime_consuming\telapsed_seconds\tstatus\texit_code\tfinished_at\tresult_path\toutdir\tnlive\n' >> "$SUMMARY_FILE"
        fi
        # Strip embedded tabs/newlines so every invocation occupies exactly one row.
        fields=("${SLURM_JOB_ID:-manual}" "$task" "$sampled" "$duration" "$elapsed" "$status" "$code" "$(date -Is)" "${RESULT:-unknown}" "${OUTDIR:-unknown}" "${NLIVE:-250}")
        for i in "${!fields[@]}"; do
            value="${fields[$i]//$'\t'/ }"
            fields[$i]="${value//$'\n'/ }"
        done
        (IFS=$'\t'; printf '%s\n' "${fields[*]}") >> "$SUMMARY_FILE"
    ) 9>"$SUMMARY_FILE.lock"
    if [[ $? != 0 ]]; then
        echo "WARNING: could not append task summary to $SUMMARY_FILE" >&2
    else
        echo "Task summary appended to $SUMMARY_FILE"
    fi
    exit "$code"
}
trap record_task_summary EXIT

source "$BASE_DIR/env.sh"
cd "$BASE_DIR"
POSTERIORS="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors"
RESULT="${RESULT:-$POSTERIORS/ET/single/seed0053_dprior.hdf5}"
OUTDIR="${OUTDIR:-bilby_injections/ET_$(basename "${RESULT%.hdf5}")_chi}"
# Keep checkpoint checks and the sampler on the same suffix-free directory.
OUTDIR="${OUTDIR%/}"
OUTDIR="${OUTDIR%.hdf5}"
PRIOR="${PRIOR:-ET}"
SPIN_MODE="${SPIN_MODE:-full}"
# Which parameters stay free. Empty = whatever --spin-mode decides. Everything not
# listed that has an injected truth gets pinned at that truth, so a short list is a
# targeted study rather than a baseline. Example:
FREE_PARAMS="${FREE_PARAMS:-}"


# ------------------------------------------------------------
# Check input
# ------------------------------------------------------------

if [ ! -f "$RESULT" ]; then
    echo "ERROR: no such injection: $RESULT" >&2
    exit 1
fi
mkdir -p "$OUTDIR"

# dynesty finds bilby_resume.pickle by outdir+label and resumes it WITHOUT checking
# that its dimensionality matches the current prior. That is how 13287046 died: it
# reloaded a finished 5-parameter run, then tried to label 5 columns with 15 names.
# Stamp what each run sampled and refuse to resume across a change.
RUN_STAMP="$OUTDIR/.free_params"
THIS_RUN="prior_schema=metadata_v1 prior=$PRIOR spin_mode=$SPIN_MODE free_params=$FREE_PARAMS"
if [ -f "$OUTDIR/bilby_resume.pickle" ]; then
    PREV_RUN="$(cat "$RUN_STAMP" 2>/dev/null || echo unknown)"
    if [ "$PREV_RUN" != "$THIS_RUN" ]; then
        echo "ERROR: $OUTDIR holds sampler state from a different parameter set." >&2
        echo "         previous: $PREV_RUN" >&2
        echo "         this job: $THIS_RUN" >&2
        echo "       dynesty would resume it and fail on the parameter count." >&2
        echo "       Clear the state (the result file is left alone):" >&2
        echo "         rm -f $OUTDIR/bilby_resume.pickle $OUTDIR/bilby_dynesty.pickle" >&2
        echo "       or point OUTDIR at a fresh directory." >&2
        exit 1
    fi
fi
echo "$THIS_RUN" > "$RUN_STAMP"

# ------------------------------------------------------------
# Preflight check
#
# - environment
# - waveform metadata
# - context data
# - injected truth inside chosen prior
# ------------------------------------------------------------

python - "$RESULT" "$PRIOR" <<'PY'

import sys
import numpy as np
import bilby
import sklearn  # noqa: F401

from dingo.gw.result import Result

sys.path.insert(
    0,
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline",
)

from builders import build_prior

assert "anaconda" not in sys.executable.lower(), sys.executable
result_path, prior_name = sys.argv[1], sys.argv[2]
r = Result(file_name=result_path)
assert "waveform" in r.context
assert "asds" in r.context
approximant = (
    r.base_metadata[
        "dataset_settings"
    ][
        "waveform_generator"
    ][
        "approximant"
    ]
)
print(
    f"preflight: detectors={r.interferometers}, "
    f"f_min={r.domain.f_min}, "
    f"f_max={r.domain.f_max}, "
    f"approximant={approximant}"
)

priors = build_prior(
    r,
    prior=prior_name,
)


# ------------------------------------------------------------
# Injection truth check
# ------------------------------------------------------------

import h5py

with h5py.File(result_path, "r") as f:

    if "injection_parameters" not in f:
        print(
            "preflight: no injection_parameters; "
            "skipping truth check"
        )
        raise SystemExit(0)

    truth = {
        k: float(
            f[f"injection_parameters/{k}"][()]
        )
        for k in f["injection_parameters"]
    }

# Dingo geocent_time is relative to t_ref.
# Bilby prior uses absolute GPS time.

if "geocent_time" in truth:
    truth["geocent_time"] += r.t_ref

outside = []

for name, value in truth.items():

    p = priors.get(name)

    if p is None:
        continue
    if not hasattr(p, "ln_prob"):
        continue
    if not np.isfinite(p.ln_prob(value)):
        minimum = getattr(p, "minimum", np.nan)
        maximum = getattr(p, "maximum", np.nan)
        outside.append(
            f"{name}={value:.6g} "
            f"not in [{minimum:.6g}, {maximum:.6g}]"
        )

if outside:
    raise SystemExit(
        f"preflight FAILED: prior '{prior_name}' "
        "excludes the injected truth:\n  "
        + "\n  ".join(outside)
    )

print(
    f"preflight OK: truth inside "
    f"the '{prior_name}' prior"
)

PY


# ------------------------------------------------------------
# Run configuration
# ------------------------------------------------------------

echo "result -> $RESULT"
echo "outdir -> $OUTDIR"
echo "spin_mode   -> $SPIN_MODE"
echo "free_params -> ${FREE_PARAMS:-(none, using spin_mode)}"
echo "prior  -> $PRIOR"


# ------------------------------------------------------------
# Bilby / dynesty run
# ------------------------------------------------------------

RUN_LOG="$OUTDIR/run_${SLURM_JOB_ID:-manual}_$start_time.log"
python -u run_baseline.py \
    --result "$RESULT" \
    --outdir "$OUTDIR" \
    --label bilby \
    --prior "$PRIOR" \
    --spin-mode "$SPIN_MODE" \
    --free-params "$FREE_PARAMS" \
    --nlive "${NLIVE:-250}" \
    --naccept "${NACCEPT:-60}" \
    --dlogz "${DLOGZ:-0.1}" \
    --npool "${SLURM_CPUS_PER_TASK:-8}" | tee "$RUN_LOG"


# ------------------------------------------------------------
# Timing
# ------------------------------------------------------------

end_time=$(date +%s)
time_taken=$((end_time - start_time))
echo "Done: $(date)"
printf "Elapsed time: %02d:%02d:%02d\n" \
    $((time_taken / 3600)) \
    $(((time_taken % 3600) / 60)) \
    $((time_taken % 60))