#!/bin/bash
#SBATCH --job-name=train_ET_5M_complete_resume
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=150:00:00
#SBATCH --mem=256G

# Resume job 13437013's training after it has stopped:
#   sbatch ET_resume_submission.sh
# Optional explicit checkpoint (must belong to TRAIN_DIR):
#   sbatch ET_resume_submission.sh training/main_train_dir_ET_5M_complete/model_150.pt
# To queue a continuation now, without overlapping the original job:
#   sbatch --dependency=afterany:13437013 ET_resume_submission.sh
# Training settings come from the checkpoint; local_settings.yaml is read from
# TRAIN_DIR. No GPU-dependent checkpoint, batch-size, or stage changes are made.

set -eo pipefail
EXAMPLE_DIR=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model
cd "$EXAMPLE_DIR"
TRAIN_DIR="${TRAIN_DIR:-$EXAMPLE_DIR/training/main_train_dir_ET_5M_complete}"
TRAIN_DIR=$(realpath "$TRAIN_DIR")
CKPT="${1:-$TRAIN_DIR/model_latest.pt}"
CKPT=$(realpath -m "$CKPT")

# Do not write to the original run while it is still active.
SOURCE_JOB_ID="${SOURCE_JOB_ID:-13437013}"
ACTIVE=$(squeue --noheader --user="$USER" --format='%A')
if printf '%s\n' "$ACTIVE" | grep -qx "$SOURCE_JOB_ID"; then
    echo "ERROR: source job $SOURCE_JOB_ID is still active. Submit after it stops, or use --dependency=afterany:$SOURCE_JOB_ID." >&2
    exit 1
fi
if [[ "$(dirname "$CKPT")" != "$TRAIN_DIR" ]]; then
    echo "ERROR: checkpoint must belong to TRAIN_DIR=$TRAIN_DIR" >&2
    exit 1
fi
test -s "$CKPT" || { echo "ERROR: missing checkpoint $CKPT" >&2; exit 1; }
test -s "$TRAIN_DIR/local_settings.yaml" || { echo "ERROR: missing local_settings.yaml" >&2; exit 1; }
# Prevent two copies of this resume script from writing to the same run.
exec 9>"$TRAIN_DIR/.resume.lock"
flock -n 9 || { echo "ERROR: another resume holds the training-directory lock" >&2; exit 1; }

module load GCCcore/12.3.0 Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages"
export PYTHONUNBUFFERED=1

echo "HOSTNAME=$(hostname)"
echo "TRAIN_DIR=$TRAIN_DIR"
echo "CKPT=$CKPT"
# Load once to detect a truncated checkpoint before changing history or backups.
RESUME_EPOCH=$(python - "$CKPT" <<'PY'
import sys
import torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
epoch = int(d["epoch"])
training = d["metadata"]["train_settings"]["training"]
completed = (d["metadata"].get("stage_progress") or {}).get("end_epochs")
if completed is None:
    raise SystemExit("ERROR: missing recorded stage boundaries; inspect the checkpoint before resuming.")
stages = []
while f"stage_{len(stages)}" in training:
    stages.append(training[f"stage_{len(stages)}"])
if len(completed) >= len(stages):
    print("All training stages are already complete.", file=sys.stderr)
    print("COMPLETE")
    raise SystemExit(0)
stage_start = completed[-1] if completed else 0
if epoch < stage_start:
    raise SystemExit("ERROR: checkpoint epoch precedes its recorded stage boundary.")
if epoch > stage_start:
    if "optimizer_state_dict" not in d:
        raise SystemExit("ERROR: missing optimizer state for mid-stage resume.")
    if d.get("scheduler_kwargs") and "scheduler_state_dict" not in d:
        raise SystemExit("ERROR: missing scheduler state for mid-stage resume.")
stage = stages[len(completed)]
print(f"Checkpoint OK: epoch={epoch}, stage={len(completed)}, "
      f"batch_size={stage['batch_size']}, configured stage end={stage_start + stage['epochs']}",
      file=sys.stderr)
print(epoch)
PY
)
if [[ "$RESUME_EPOCH" == COMPLETE ]]; then
    exit 0
fi

STAMP="$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-manual}"
# EarlyStopping is recreated by the existing training code on resume.
# Preserve the previous best before its first resumed epoch can replace it.
if [[ -f "$TRAIN_DIR/best_model.pt" ]]; then
    cp -p "$TRAIN_DIR/best_model.pt" "$TRAIN_DIR/best_model_before_resume_$STAMP.pt"
fi
# History can be ahead of the last saved checkpoint after an interrupted write.
HIST="$TRAIN_DIR/history.txt"
if [[ -s "$HIST" ]]; then
    LAST=$(awk 'NF {last=$1} END {print last}' "$HIST")
    if [[ "$LAST" =~ ^[0-9]+$ ]] && (( LAST > RESUME_EPOCH )); then
        cp -p "$HIST" "$TRAIN_DIR/history_before_resume_$STAMP.txt"
        awk -v e="$RESUME_EPOCH" '$1 <= e' "$HIST" > "$HIST.resume.tmp"
        mv "$HIST.resume.tmp" "$HIST"
        echo "History rewound from epoch $LAST to $RESUME_EPOCH; original backed up."
    fi
fi
dingo_train --checkpoint "$CKPT" --train_dir "$TRAIN_DIR"
echo "Resume job finished."
