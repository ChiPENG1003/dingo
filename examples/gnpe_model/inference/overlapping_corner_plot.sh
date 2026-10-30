#!/bin/bash
#SBATCH --job-name=overlap_corner
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%A.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=00:20:00
#SBATCH --mem=8G

# Overlay the overlap posteriors (different Delta t) on ONE corner via -c.
#
# The file list used to be spelled out as injection_overlap_<shift>s_80mpc.hdf5,
# but only the 0.05 s run ever had that name -- the others are plain
# injection_overlap_<shift>.hdf5, so every path but one missed. Globbing the
# directory keeps this working as the naming evolves.

set -eo pipefail

GNPE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
LOG_DIR="$GNPE_DIR/logs/dingo"
mkdir -p "$LOG_DIR"
cd "$GNPE_DIR/inference"

echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

module load GCCcore/12.3.0
module load Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages"

# Prefer results from the current layout; fall back to the pre-reorganisation ones.
mapfile -t OVERLAPS < <(
    find inference_result/posteriors inference_result/legacy/overlap \
         -name '*overlap*.hdf5' -o -name '*_shift*.hdf5' 2>/dev/null | sort
)
if [ ${#OVERLAPS[@]} -eq 0 ]; then
    echo "ERROR: no overlap posteriors found under inference_result/" >&2
    exit 1
fi
echo "Overlaying ${#OVERLAPS[@]} posteriors:"
printf '  %s\n' "${OVERLAPS[@]}"

python inference_result/analysis/plot_single_corner.py \
  -c "${OVERLAPS[@]}" \
  -p chirp_mass mass_ratio luminosity_distance theta_jn ra dec psi geocent_time

echo "Done."
