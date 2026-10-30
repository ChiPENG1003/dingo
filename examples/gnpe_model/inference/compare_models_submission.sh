#!/bin/bash
#SBATCH --job-name=ET_compare_models
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

set -euo pipefail

mkdir -p "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

# This script lives in gnpe_model/inference/. Work from that directory so the
# relative paths below (inference.py, inference_result/...) resolve, whichever
# directory sbatch was invoked from.
cd "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference"

module load GCCcore/12.3.0
module load Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:${PYTHONPATH:-}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODE=compare
export SINGLE_SEED=53
export NUM_SAMPLES=500

# First entry is the reference: it fixes the truth, the prior and the noise draw.
export COMPARE_MODELS="ET_200,ET_150,ET_100, ET_latest"

# 0 => luminosity_distance comes from the model prior (stays in trained support).
export FIX_DISTANCE=0

python inference.py

python inference_result/analysis/plot_posteriors.py \
    --dingo \
        inference_result/posteriors/compare/seed0053_dprior/ET_200.hdf5 \
        inference_result/posteriors/compare/seed0053_dprior/ET_150.hdf5 \
        inference_result/posteriors/compare/seed0053_dprior/ET_100.hdf5 \
        inference_result/posteriors/compare/seed0053_dprior/ET_latest.hdf5 \
    --bilby none \
    -p chirp_mass mass_ratio luminosity_distance chi_eff

echo "Done."
