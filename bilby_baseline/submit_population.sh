#!/bin/bash
#SBATCH --job-name=bilby_ET_seed0053
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=public-cpu
#SBATCH --time=4-00:00:00
#SBATCH --mem=16G

# dynesty on ONE Dingo injection file. Everything that changes between runs is
# an environment variable, and the SBATCH lines above are overridable on the
# command line -- a bigger run does not need its own copy of this script:
#
#   sbatch submit_population.sh
#   INJ_NAME=01_reference.hdf5 INJ_DIR=.../series NLIVE=1000 NACCEPT=60 DLOGZ=0.1 \
#       sbatch --cpus-per-task=16 --time=4-00:00:00 --mem=16G submit_population.sh
#
# npool follows --cpus-per-task, so the pool never oversubscribes the allocation.

set -e

start_time=$(date +%s)
mkdir -p logs
echo "HOSTNAME=$(hostname)  $start_time"

BASE_DIR="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline"
source "$BASE_DIR/env.sh"
cd "$BASE_DIR"

# Keep ET results separate from previous runs and inherited shell names.
INJ_NAME="${INJ_NAME:-seed0053_dprior.hdf5}"
PRIOR="${PRIOR:-ET}"
OUTDIR="${OUTDIR:-$BASE_DIR/bilby_injections/${OUT_NAME:-${PRIOR}_${INJ_NAME%.hdf5}}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
printf 'prior -> %s\noutdir -> %s\nnpool -> %s\n' "$PRIOR" "$OUTDIR" "${SLURM_CPUS_PER_TASK:-4}"

python -u run_population.py \
    --nlive "${NLIVE:-500}" \
    --inj_dir "${INJ_DIR:-/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors/ET/single}" \
    --inj_name "$INJ_NAME" \
    --outdir "$OUTDIR" \
    --naccept "${NACCEPT:-20}" \
    --dlogz "${DLOGZ:-0.5}" \
    --npool "${SLURM_CPUS_PER_TASK:-4}" \
    --prior "$PRIOR"

end_time=$(date +%s)
time_taken=$((end_time-start_time))
echo "Done. $end_time"

printf "Elapsed time: %02d:%02d:%02d\n" \
    $((time_taken / 3600)) \
    $(((time_taken % 3600) / 60)) \
    $((time_taken % 60))
