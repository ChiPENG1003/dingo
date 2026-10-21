#!/bin/bash
#SBATCH --job-name=bn_probe
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/bn_frozen_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/bn_frozen_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=shared-cpu
#SBATCH --time=01:00:00
#SBATCH --mem=64G

set -eo pipefail
cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
export PYTHONUNBUFFERED=1
module purge
module load GCCcore/12.3.0 Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages

SCR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/diagnostics
echo "############ train-vs-eval probe ############"
python -u $SCR/bn_probe.py training/main_train_dir_ET/model_stage_0.pt cpu
echo
echo
echo "############ loss decomposition, RB still frozen ############"
python -u $SCR/loss_decompose.py training/main_train_dir_ET/model_stage_0.pt stage_0
