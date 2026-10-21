#!/bin/bash
#SBATCH --job-name=multi_ET
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-gpu,shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE|COMPUTE_TYPE_ADA|COMPUTE_TYPE_HOPPER|COMPUTE_TYPE_BLACKWELL"
#SBATCH --time=08:00:00
#SBATCH --mem=64G

# Population run for the ET GNPE model: N injections drawn from the training
# prior, one posterior each, for the coverage / ratio-histogram analysis.
#
#   sbatch multi_ET_submission.sh
#   NUM_INJECTIONS=500 NUM_SAMPLES=5000 sbatch multi_ET_submission.sh
#
# Resumable: `multi` skips any injection_NNNN.hdf5 that already exists, so a
# re-submit after a timeout picks up where the last one stopped.
#
# The checkpoint is PINNED to model_200.pt rather than the registry's ET entry
# (model_latest.pt): the ET main training is still running and rewrites
# model_latest.pt (4.4 GB) every epoch, so loading it can catch a half-written
# file. The label is still "ET", so results land in posteriors/ET/multi/.

set -eo pipefail

mkdir -p "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo"
echo "HOSTNAME=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# Without the --constraint above, a Pascal node (sm_60) makes every injection
# raise CUBLAS_STATUS_ARCH_MISMATCH. Fail loudly instead of skipping all 500.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
if [ "$(echo "$CC" | cut -d. -f1)" -lt 7 ]; then
    echo "ERROR: GPU compute capability $CC is below the sm_75 this PyTorch supports." >&2
    exit 1
fi

cd "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference"

module load GCCcore/12.3.0 Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODEL_SPEC="${MODEL_SPEC:-ET=main_train_dir_ET/model_latest.pt|init_train_dir_ET/model_latest.pt}"

# `multi` only ever ADDS to a population: it skips every index that already has a
# file. Re-running a retrained network into a populated directory therefore
# produces nothing at all and still exits 0 (cost job 12655139, 2026-09-11).
# Give a new network its own OUT_NAME.
OUT_NAME="${OUT_NAME:-}"
if [ -z "$OUT_NAME" ]; then
    echo "NOTE: OUT_NAME unset -> writing to the default 'multi' directory;" \
         "injections already there will be SKIPPED, not regenerated." >&2
fi

start_time=$(date +%s)
python -u inference.py multi \
    --model "$MODEL_SPEC" \
    ${OUT_NAME:+--out-name "$OUT_NAME"} \
    --num-injections "${NUM_INJECTIONS:-500}" \
    --base-seed "${BASE_SEED:-0}" \
    --num-samples "${NUM_SAMPLES:-5000}" \
    --batch-size "${BATCH_SIZE:-5000}" \
    --num-iterations "${NUM_ITERATIONS:-30}"

end_time=$(date +%s)
printf "Done. Elapsed %02d:%02d:%02d\n" \
    $(( (end_time-start_time) / 3600 )) \
    $(( ((end_time-start_time) % 3600) / 60 )) \
    $(( (end_time-start_time) % 60 ))
