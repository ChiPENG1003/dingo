#!/bin/bash
#SBATCH --job-name=train_JPE_main_2M
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=6-06:00:00
#SBATCH --mem=128G

set -eo pipefail
cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
TRAIN_DIR=training/main_train_dir_ET_JPE_2M
test -s training_data/waveform_dataset_ET_joint_separate_2M.hdf5
test ! -e "$TRAIN_DIR/model_latest.pt" || { echo "Existing main checkpoint; refusing fresh start" >&2; exit 1; }
mkdir -p logs/dingo /tmp/dingo_jpe_main "$TRAIN_DIR"
module purge
module load GCCcore/12.3.0 Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
echo "Starting JPE main model from scratch on $(hostname)"
dingo_train --settings_file configs/train_settings_ET_JPE.yaml --train_dir "$TRAIN_DIR"
