#!/bin/bash
#SBATCH --job-name=pp_compare100
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=02:00:00
#SBATCH --mem=32G

set -eo pipefail
ROOT=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo
RESULT_ROOT="$ROOT/examples/gnpe_model/inference/inference_result"
cd "$RESULT_ROOT/analysis"
module load GCCcore/12.3.0 Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH="$ROOT:$HOME/.local/lib/python3.11/site-packages"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONDONTWRITEBYTECODE=1
python -u pp_plot.py --paired-manifest pp_pairs100_20261023.csv --comparison-outdir ../figures/PP_Dingo_Bilby_100_20261023
