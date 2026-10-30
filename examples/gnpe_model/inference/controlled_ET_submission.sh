#!/bin/bash
#SBATCH --job-name=ctrl_ET
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-gpu,shared-gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="COMPUTE_TYPE_TURING|COMPUTE_TYPE_AMPERE|COMPUTE_TYPE_ADA|COMPUTE_TYPE_HOPPER|COMPUTE_TYPE_BLACKWELL"
#SBATCH --time=08:00:00
#SBATCH --mem=64G

# Two CONTROLLED variants of the ET population run, each isolating one of the
# two random sources. The plain run (posteriors/ET/multi/) varies both.
#
#   EXPERIMENT=vary_noise sbatch controlled_ET_submission.sh
#       truth pinned to seed0053_dprior.hdf5, 500 Gaussian noise realizations.
#       The ET ASD dataset holds a single design curve, so "varying the ASD"
#       is exactly varying the noise drawn from it:
#       noise = (randn + i*randn) * asd * noise_std.
#
#   EXPERIMENT=vary_theta sbatch controlled_ET_submission.sh
#       500 truths from the prior, ONE noise realization reused for all of them.
#
# Results go to posteriors/ET/controlled/<experiment>/ -- beside, never on top
# of, posteriors/ET/multi/.
#
# Same pinned checkpoint as multi_ET_submission.sh (model_200.pt, not
# model_latest.pt, which the running training job rewrites every epoch), so the
# three runs are comparable.

set -eo pipefail

mkdir -p "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo"
echo "HOSTNAME=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || true

# The --constraint above should already exclude the Pascal nodes, whose CC 6.x
# this PyTorch build has no kernels for. Check anyway: without it every single
# injection raises CUBLAS_STATUS_ARCH_MISMATCH, and a job that skips 500 bad
# injections still exits 0. Cost job 12521294.
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

MODEL_SPEC="${MODEL_SPEC:-ET=main_train_dir_ET/model_200.pt|init_train_dir_ET/model_latest.pt}"
REFERENCE="${REFERENCE:-inference_result/posteriors/ET/single/seed0053_dprior.hdf5}"
EXPERIMENT="${EXPERIMENT:?set EXPERIMENT=vary_noise or EXPERIMENT=vary_theta}"

case "$EXPERIMENT" in
    vary_noise) CONTROL=(--fixed-truth "$REFERENCE") ;;
    vary_theta) CONTROL=(--fixed-noise-seed "${NOISE_SEED:-20250908}") ;;
    *) echo "ERROR: EXPERIMENT must be vary_noise or vary_theta, got '$EXPERIMENT'" >&2
       exit 1 ;;
esac

start_time=$(date +%s)
python -u inference.py multi \
    --model "$MODEL_SPEC" \
    --out-name "controlled/$EXPERIMENT" \
    "${CONTROL[@]}" \
    --num-injections "${NUM_INJECTIONS:-500}" \
    --base-seed "${BASE_SEED:-53}" \
    --num-samples "${NUM_SAMPLES:-500}" \
    --batch-size "${BATCH_SIZE:-5000}" \
    --num-iterations "${NUM_ITERATIONS:-30}"

end_time=$(date +%s)
printf "Done. Elapsed %02d:%02d:%02d\n" \
    $(( (end_time-start_time) / 3600 )) \
    $(( ((end_time-start_time) % 3600) / 60 )) \
    $(( (end_time-start_time) % 60 ))
