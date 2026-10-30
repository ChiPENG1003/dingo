#!/bin/bash
#SBATCH --job-name=generate_waveform_dataset
#SBATCH --output=logs/dingo/%x_%j.out
#SBATCH --error=logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=private-dpnc-gpu
#SBATCH --gres=gpu:nvidia_rtx_a5000:1
#SBATCH --time=12:00:00
#SBATCH --mem=128G

mkdir -p logs/dingo
echo "HOSTNAME=$(hostname)"
echo "PWD=$(pwd)"

module load GCCcore/12.3.0
module load Python/3.11.3
source $HOME/venvs/dingo_env/bin/activate

cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/npe_model

echo "=== waveform_dataset_settings.yaml ===="
cat waveform_dataset_settings.yaml
echo "======================================="

dingo_train --settings_file train_settings.yaml --train_dir training


#dingo_generate_dataset --settings waveform_dataset_settings.yaml --out_file training_data/waveform_dataset.hdf5
#dingo_generate_asd_dataset --settings_file asd_dataset_settings.yaml --data_dir training_data/asd_dataset

#cd /srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/npe_model/training_data

# mv waveform_dataset.hdf5 waveform_dataset_BNS_3.hdf5

echo "Done."