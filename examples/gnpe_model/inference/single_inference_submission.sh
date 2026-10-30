#!/bin/bash
#SBATCH --job-name=inference_ET_org
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G

# ----------------------------- WHAT TO RUN -----------------------------------
# single         one injection
# overlap        one injection plus a second signal shifted in time
# multi          many injections (population / coverage)
# overlap_multi  many two-signal injections
# real           observed strain for a real event
# compare        several models on ONE injection
MODE=${MODE:-single}

# O2-ASD injection: inference.py reads this checkpoint's stage-1
# ``training_data/asd_dataset/asds_O2.hdf5`` automatically.
# List registered models and their ASD source with: python inference.py models
MODEL=${MODEL:-ET_org}

NUM_SAMPLES=${NUM_SAMPLES:-500}
BATCH_SIZE=${BATCH_SIZE:-${NUM_SAMPLES}}
NUM_ITERATIONS=${NUM_ITERATIONS:-30}     # GNPE Gibbs iterations

# ---------------------- TARGET: which injection / event ----------------------
# single / overlap / compare -- the seed IS the injection: it fixes both the
# truth parameters and the noise realization, and it names the output file.
# Seed 0 already exists as seed0000_dprior.hdf5, so default to a new injection.
SEED=${SEED:-53}

# 0 keeps luminosity_distance from the model's own prior, Uniform(100, 2000) Mpc.
# A non-zero value draws from N(DISTANCE, DISTANCE_STD) instead, which at the
# old default of 80 lands ~77% of injections BELOW the prior minimum, i.e.
# outside the range the network was trained on.
DISTANCE=${DISTANCE:-0}
DISTANCE_STD=${DISTANCE_STD:-25}

# overlap / overlap_multi -- how much later the second signal merges, seconds.
SHIFT=${SHIFT:-0.05}

# multi / overlap_multi -- injection i uses seed BASE_SEED + i.
NUM_INJECTIONS=${NUM_INJECTIONS:-1000}
BASE_SEED=${BASE_SEED:-0}
EVENT=${EVENT:-GW170809}

BUILD_EVENT=${BUILD_EVENT:-no} # can use "auto" to build the event if missing.

# compare -- the FIRST entry is the reference: it fixes the truth, the prior and
COMPARE_MODELS=${COMPARE_MODELS:-300SVD,200SVD,50k300SVD}
# =============================================================================

set -eo pipefail

GNPE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
mkdir -p "$GNPE_DIR/logs/dingo"
cd "$GNPE_DIR/inference"

case "$MODE" in
    single|overlap|multi|overlap_multi|real|compare) ;;
    *)
        echo "ERROR: MODE=$MODE is not valid." >&2
        echo "       Use one of: single overlap multi overlap_multi real compare" >&2
        exit 1
        ;;
esac

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

module load GCCcore/12.3.0
module load Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# inference.py reads these; only the ones the chosen mode uses have any effect.
export MODE MODEL
export NUM_SAMPLES BATCH_SIZE NUM_ITERATIONS
export SINGLE_SEED="$SEED"
export FIX_DISTANCE="$DISTANCE"
export FIX_DISTANCE_STD="$DISTANCE_STD"
export SHIFT_SECONDS="$SHIFT"
export NUM_INJECTIONS BASE_SEED
export COMPARE_MODELS

echo "=============================================================="
echo " MODE=$MODE  MODEL=$MODEL  NUM_SAMPLES=$NUM_SAMPLES"
case "$MODE" in
    single)        echo " target: injection seed=$SEED, distance=$DISTANCE" ;;
    overlap)       echo " target: injection seed=$SEED, distance=$DISTANCE, shift=${SHIFT}s" ;;
    multi)         echo " target: $NUM_INJECTIONS injections, seeds ${BASE_SEED}+i" ;;
    overlap_multi) echo " target: $NUM_INJECTIONS injections, seeds ${BASE_SEED}+i, shift=${SHIFT}s" ;;
    real)          echo " target: real event $EVENT" ;;
    compare)       echo " target: injection seed=$SEED across $COMPARE_MODELS" ;;
esac
echo "=============================================================="

if [ "$MODE" = "real" ]; then
    EVENT_FILE="inference_result/events/${EVENT}.hdf5"
    if [ "$BUILD_EVENT" = "yes" ] || { [ "$BUILD_EVENT" = "auto" ] && [ ! -f "$EVENT_FILE" ]; }; then
        echo "Building context for $EVENT ..."
        # Conditioning downloads strain from GWOSC. Compute nodes may have no
        # outbound network, in which case run this one command on the login node
        # and resubmit with BUILD_EVENT=no.
        python "$GNPE_DIR/construct_plot_data/construct_real_event.py" --event "$EVENT"
    fi
    if [ ! -f "$EVENT_FILE" ]; then
        echo "ERROR: $EVENT_FILE not found. Build it on the login node with:" >&2
        echo "  python $GNPE_DIR/construct_plot_data/construct_real_event.py --event $EVENT" >&2
        exit 1
    fi
    export REAL_EVENT_FILE="$EVENT_FILE"
fi

# Forward explicit CLI options, e.g. single --model ET_chi --allow-incomplete-gnpe.
python inference.py "$@"

echo "Done."
