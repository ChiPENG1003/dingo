#!/bin/bash
#SBATCH --job-name=injection_combo
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%A_%a.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared-gpu,private-dpnc-gpu
# VramPerGpu is declared no_consume on this cluster, so it SELECTS a card but
# does not reserve its memory: several array tasks land on the same GPU and
# share it. Each task holds ~2.5 GB, so a bare --gres=gpu:1 lets four of them
# onto a 10 GB RTX 3080 and the fourth dies with CUDA OOM at model load
# (9 tasks of array 11522789 died exactly this way). Asking for 20G keeps the
# array on the 24 GB+ cards, where ~9 co-scheduled tasks still fit.
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time=02:00:00
#SBATCH --mem=32G

# ---------------------------------------------------------------------------
# One array task per grid cell.
#
#   sbatch --array=0-99 submit_grid.sh snr_ratio
#   sbatch --array=0-99 submit_grid.sh chirp_mass
#   sbatch --array=0-99 submit_grid.sh mass_ratio
#   sbatch --array=0-99 submit_grid.sh luminosity_distance
#   sbatch            submit_grid.sh reference      # control, no array
#
# CELLS_PER_TASK=10 with --array=0-9 gives one task per column instead, which
# trades queue throughput for fewer model loads.
#
# Re-running is safe and is the intended way to finish a partial sweep: cells
# skip the realisations they already hold.
# ---------------------------------------------------------------------------

AXIS=${1:-${AXIS:-snr_ratio}}

MODEL=${MODEL:-1M_7params_O2}
TARGET_SEED=${TARGET_SEED:-1}          # select_target.py reports which seeds work
TARGET_SNR1=${TARGET_SNR1:-25}
NUM_REALIZATIONS=${NUM_REALIZATIONS:-100}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
NUM_ITERATIONS=${NUM_ITERATIONS:-30}
CELLS_PER_TASK=${CELLS_PER_TASK:-1}

set -eo pipefail

GNPE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
COMBO_DIR="$GNPE_DIR/inference/injection_combo"
mkdir -p "$GNPE_DIR/logs/dingo"
cd "$COMBO_DIR"

echo "HOSTNAME=$(hostname)"
echo "AXIS=$AXIS  TASK=${SLURM_ARRAY_TASK_ID:-none}  CELLS_PER_TASK=$CELLS_PER_TASK"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load GCCcore/12.3.0
module load Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# construct_data.py imports pyplot for the SNR helper; no display on a node.
export MPLBACKEND=Agg

COMMON=(--model "$MODEL"
        --target-seed "$TARGET_SEED"
        --target-snr1 "$TARGET_SNR1"
        --num-realizations "$NUM_REALIZATIONS"
        --num-samples "$NUM_SAMPLES"
        --num-iterations "$NUM_ITERATIONS")

if [ "$AXIS" = "reference" ]; then
    python run_grid.py --reference "${COMMON[@]}"
else
    python run_grid.py "$AXIS" --cells-per-task "$CELLS_PER_TASK" "${COMMON[@]}"
fi

echo "Done."
