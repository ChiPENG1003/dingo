# Sourced by submit_*.sh — sets up a SELF-CONTAINED Dingo/bilby runtime on
# Baobab. Must not depend on the submitting shell's environment (sbatch may
# inherit an Anaconda base env whose pyarrow/numpy2 ABI is broken).

# Start from a clean module + path state so nothing leaks in from the caller.
module purge 2>/dev/null || true
module load GCCcore/12.3.0
module load Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"

# Clean PYTHONPATH: ONLY the repo + the user site-packages. Do NOT append the
# inherited $PYTHONPATH — that is how Anaconda's pyarrow leaks in and crashes
# sklearn with "numpy.core.multiarray failed to import".
export PYTHONPATH="/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo:$HOME/.local/lib/python3.11/site-packages"

# Belt-and-suspenders: drop any Anaconda dirs from PATH.
export PATH="$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -i 'anaconda' | paste -sd: -)"
unset PYTHONHOME 2>/dev/null || true
