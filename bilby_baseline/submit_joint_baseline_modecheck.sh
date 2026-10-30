#!/bin/bash
#SBATCH --job-name=bilby_joint_modecheck
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/logs/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=private-dpnc-cpu
#SBATCH --time=6-06:00:00
#SBATCH --mem=64G

set -eo pipefail
BASE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline
RESULT=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/inference/inference_result/posteriors/ET_5M_complete_best140/overlap/seed10051_dprior_shift0.100s_bilby_joint.hdf5
OUTDIR=$BASE_DIR/bilby_injections/ET5M_best140_joint_seed10051_shift0p1_nlive250_actwalk

source "$BASE_DIR/env.sh"
cd "$BASE_DIR"
mkdir -p "$OUTDIR" logs

python -u run_baseline.py \
    --result "$RESULT" \
    --outdir "$OUTDIR" \
    --label bilby_joint_actwalk \
    --prior ET \
    --joint-mode joint \
    --spin-mode full \
    --nlive 250 \
    --sample act-walk \
    --nact 5 \
    --maxmcmc 20000 \
    --dlogz 0.1 \
    --npool "${SLURM_CPUS_PER_TASK:-8}"
