#!/bin/bash
#SBATCH --job-name=train_ET_5M_complete
#SBATCH --output=logs/dingo/%x_%j.out
#SBATCH --error=logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --mem=256G

set -eo pipefail

mkdir -p logs/dingo
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

export PYTHONUNBUFFERED=1

module load GCCcore/12.3.0
module load Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate

# --- 1. ASD dataset -----------------------------------------------------------
# dingo_generate_asd_dataset \
#     --settings_file configs/asd_dataset_settings_ET.yaml \
#     --data_dir training_data/asd_dataset_ET \
#     --out_name training_data/asd_dataset_ET/asds_ET_100.hdf5

# --- 2. Waveform dataset ------------------------------------------------------
# dingo_generate_dataset \
#     --settings_file configs/waveform_dataset_settings_ET.yaml \
#     --num_processes 16 \
#     --out_file training_data/waveform_dataset_ET_1M.hdf5

# --- 3. GNPE initialization network -------------------------------------------
# dingo_train --settings_file configs/train_settings_init_ET.yaml --train_dir training/init_train_dir_ET_1M

# echo " "
# echo "=============================================="
# echo "=== Finished the initial training step. ==="
# echo "=============================================="
# echo " "

# --- 4. Main GNPE network -----------------------------------------------------
echo "=============================================="
echo "Start main training step..."
echo "=============================================="
echo " "
dingo_train --settings_file configs/train_settings_ET.yaml --train_dir training/main_train_dir_ET_5M_complete
echo "=============================================="
echo "=== Finished the main training step. ==="
echo "=============================================="
echo " "

echo "Done."
