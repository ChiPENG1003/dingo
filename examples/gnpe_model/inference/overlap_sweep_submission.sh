#!/bin/bash
#SBATCH --job-name=overlap_sweep
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=64G

set -eo pipefail

SHIFTS=("$@")
if [ ${#SHIFTS[@]} -eq 0 ]; then
    SHIFTS=(0.05 0.1 0.2 0.5 1.0)
fi

mkdir -p "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo"
# This script lives in gnpe_model/inference/. Work from that directory so the
# relative paths below (inference.py, inference_result/...) resolve, whichever
# directory sbatch was invoked from.
cd "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"
echo "SHIFTS=${SHIFTS[*]}"

module load GCCcore/12.3.0
module load Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages:$PYTHONPATH

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export MODE=overlap
export SINGLE_SEED=0
export NUM_SAMPLES=1000

for s in "${SHIFTS[@]}"; do
    echo "=============================================="
    echo "=== SHIFT_SECONDS=$s ==="
    echo "=============================================="
    export SHIFT_SECONDS="$s"
    python inference.py
done

echo "Done. Generated overlap injections for shifts: ${SHIFTS[*]}"
