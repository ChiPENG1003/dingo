#!/bin/bash
#SBATCH --job-name=multi_inference
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%A_%a.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G


mkdir -p "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo"
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

# This script lives in gnpe_model/inference/. Work from that directory so the
# relative paths below (inference.py, inference_result/...) resolve, whichever
# directory sbatch was invoked from.
cd "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference"

module load GCCcore/12.3.0
module load Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODE=multi
export BASE_SEED=0
export NUM_SAMPLES=250

python inference.py

echo "Done."
