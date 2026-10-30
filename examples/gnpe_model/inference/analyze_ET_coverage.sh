#!/bin/bash
#SBATCH --job-name=coverage_ET
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=04:00:00
#SBATCH --mem=16G

# Coverage + ratio histograms for the ET population run (multi_ET_submission.sh).
# The KDE mode of every parameter of every injection is the whole cost, so this
# is CPU-bound and belongs on a compute node, not the login node.
#
#   sbatch analyze_ET_coverage.sh
#   N_SAMPLES=1000 sbatch analyze_ET_coverage.sh   # subsample before reducing

set -eo pipefail

cd "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/analysis"

module load GCCcore/12.3.0 Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages

INJ_DIR="${INJ_DIR:-/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors/ET/multi}"

python -u read_data.py \
    --injections-dir "$INJ_DIR" \
    --label "${LABEL:-ET}" \
    ${PARAMS:+-p $PARAMS} \
    ${N_SAMPLES:+--n-samples "$N_SAMPLES"} \
    ${REBUILD:+--rebuild}

echo "Done."
