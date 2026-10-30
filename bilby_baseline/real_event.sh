#!/bin/bash
#SBATCH --job-name=bilby_real
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=10:00:00
#SBATCH --mem=16G

set -eo pipefail

start_time=$(date +%s)
mkdir -p logs
echo "HOSTNAME=$(hostname)  $start_time"

BASE_DIR="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline"
source "$BASE_DIR/env.sh"
cd "$BASE_DIR"

EVENT="${EVENT:-GW190814}"

# A real event has no injected truth, so all 15 parameters are sampled -- this is
# a much bigger run than the 8-D injection baseline. npool matches cpus-per-task.
python -u run_real_event.py \
    --event "$EVENT" \
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
