#!/bin/bash
#SBATCH --job-name=pp_ET
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=private-dpnc-cpu,shared-cpu
#SBATCH --time=01:00:00
#SBATCH --mem=16G

# P-P plot for the ET population run (multi_ET_submission.sh). Reads the SAME
# injection files as analyze_ET_coverage.sh -- a P-P plot needs no extra
# injections, it just evaluates at every credible level instead of only 0.90.
#
#   sbatch pp_plot_ET.sh

set -eo pipefail

cd "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/analysis"

module load GCCcore/12.3.0 Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages

INJ_DIR="${INJ_DIR:-/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors/ET/multi}"

python -u pp_plot.py \
    --injections-dir "$INJ_DIR" \
    --label "${LABEL:-ET}" \
    ${REBUILD:+--rebuild}

echo "Done."
