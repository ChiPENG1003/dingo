#!/bin/bash
#SBATCH --job-name=bilby_joint_ET5M
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/logs/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=5-00:00:00
#SBATCH --mem=32G

set -eo pipefail

BASE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline
RESULT=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors/ET_5M_complete_best140/overlap/seed10051_dprior_shift0.100s_bilby_joint.hdf5
OUTDIR=$BASE_DIR/bilby_injections/ET5M_best140_joint_seed10051_shift0p1_nlive250

source "$BASE_DIR/env.sh"
cd "$BASE_DIR"
mkdir -p "$OUTDIR" logs

python -u run_baseline.py \
    --result "$RESULT" \
    --outdir "$OUTDIR" \
    --label bilby_joint \
    --prior ET \
    --joint-mode joint \
    --spin-mode full \
    --nlive 250 \
    --naccept 60 \
    --dlogz 0.5 \
    --npool "${SLURM_CPUS_PER_TASK:-16}"
