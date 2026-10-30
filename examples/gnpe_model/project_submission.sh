#!/bin/bash
#SBATCH --job-name=main_training_chi_eff
#SBATCH --output=logs/dingo/%x_%j.out
#SBATCH --error=logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --mem=256G

mkdir -p logs/dingo
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

# BeeGFS reports st_blksize=512K, so Python sizes the stdout buffer at 512K and the
# .out file only updates every ~15 epochs, making the run look frozen. Flush per line.
export PYTHONUNBUFFERED=1

module load GCCcore/12.3.0
module load Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate

# echo "=== configs/waveform_dataset_settings.yaml ==="
# cat configs/waveform_dataset_settings.yaml
# echo "======================================="

# dingo_generate_asd_dataset --settings_file configs/asd_dataset_settings_fiducial.yaml --data_dir training_data/asd_dataset_fiducial --out_name training_data/asd_dataset_fiducial/asds_O2_fiducial.hdf5

# dingo_generate_dataset --settings configs/waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5

# dingo_generate_asd_dataset --settings_file configs/asd_dataset_settings.yaml --data_dir training_data/asd_dataset

# if [ -f training/main_train_dir/history.txt ]; then
#     echo "removing old training history..."
#     rm training/main_train_dir/history.txt
# fi

# if [ -f training/init_train_dir/history.txt ]; then
#     echo "removing old training history..."
#     rm training/init_train_dir/history.txt
# fi

# cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/training_data
# mv waveform_dataset.hdf5 waveform_dataset_1M.hdf5
# cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# dingo_train --settings_file configs/train_settings_init.yaml --train_dir training/init_train_dir_7params_O2

# echo " "
# echo "=============================================="
# echo "=== Finished the initial training step. ==="
# echo "=============================================="
# echo " "

dingo_train --settings_file configs/train_settings.yaml --train_dir training/main_train_dir_1M_chi_eff

echo "Done."
