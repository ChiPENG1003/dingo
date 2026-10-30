#!/bin/bash
#SBATCH --job-name=injection_series
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared-gpu,private-dpnc-gpu
# See submit_grid.sh: VramPerGpu selects a card without reserving it, so ask for
# a big one rather than share a 10 GB card with someone else's array.
#SBATCH --gres=gpu:1,VramPerGpu:20G
#SBATCH --time=01:00:00
#SBATCH --mem=32G

# ---------------------------------------------------------------------------
# The ten-injection series (run_series.py): one posterior per file, one model
# load for all ten. Re-running skips whatever is already on disk, so a preempted
# job just picks up where it stopped.
#
#   sbatch submit_series.sh                 # all ten
#   sbatch submit_series.sh --only 05 07    # redo two
# ---------------------------------------------------------------------------

MODEL=${MODEL:-1M_7params_O2}
TARGET_SEED=${TARGET_SEED:-1}
TARGET_SNR1=${TARGET_SNR1:-25}
NUM_SAMPLES=${NUM_SAMPLES:-1000}
NUM_ITERATIONS=${NUM_ITERATIONS:-30}

set -eo pipefail

GNPE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
COMBO_DIR="$GNPE_DIR/inference/injection_combo"
mkdir -p "$GNPE_DIR/logs/dingo"
cd "$COMBO_DIR"

echo "HOSTNAME=$(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load GCCcore/12.3.0
module load Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# construct_data.py imports pyplot for the SNR helper; no display on a node.
export MPLBACKEND=Agg

python -u run_series.py \
    --model "$MODEL" \
    --target-seed "$TARGET_SEED" \
    --target-snr1 "$TARGET_SNR1" \
    --num-samples "$NUM_SAMPLES" \
    --num-iterations "$NUM_ITERATIONS" \
    "$@"

echo "Done."
