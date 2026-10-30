#!/bin/bash
#SBATCH --job-name=ET5M_best140_corner10
#SBATCH --output=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.out
#SBATCH --error=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/gnpe_model/logs/dingo/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --mem=64G
set -eo pipefail
ROOT=/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo
cd "$ROOT/examples/gnpe_model/inference"
module load GCCcore/12.3.0 Python/3.11.3
source "$HOME/venvs/dingo_env/bin/activate"
export PYTHONPATH="$ROOT:$HOME/.local/lib/python3.11/site-packages"
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export MPLBACKEND=Agg
python -u - <<'PY'
import copy, os, time
import h5py
import numpy as np
import torch
import inference as inf
from dingo.gw.result import Result
root = os.getcwd()
label = "ET_5M_complete_best140"
main = os.path.abspath("../training/main_train_dir_ET_5M_complete/best_model.pt")
init = os.path.abspath("../training/init_train_dir_ET/model_latest.pt")
print("CHECKPOINT", inf.checkpoint_summary(main), flush=True)
main_pm, init_pm = inf._load_pair(main, init)
outdir = os.path.join(inf.POSTERIOR_DIR, label, "best10_20261021")
os.makedirs(outdir, exist_ok=True)
for i in range(10):
    name = f"injection_{i:04d}"
    src = os.path.join(inf.POSTERIOR_DIR, "ET_best/best10_20261007", name + ".hdf5")
    dst = os.path.join(outdir, name + ".hdf5")
    if os.path.exists(dst):
        raise FileExistsError(dst)
    old = Result(file_name=src)
    domain = inf.build_domain_from_model_metadata(main_pm.metadata)
    for attr in ("f_min", "f_max", "delta_f"):
        assert getattr(domain, attr) == getattr(old.domain, attr), attr
    assert main_pm.metadata["train_settings"]["data"]["ref_time"] == old.t_ref
    sampler = inf.build_sampler(main_pm, init_pm, num_iterations=30)
    sampler.context = copy.deepcopy(old.context)
    sampler.event_metadata = copy.deepcopy(old.event_metadata)
    torch.manual_seed(10000 + i)
    torch.cuda.manual_seed_all(10000 + i)
    start = time.time()
    print("START", name, flush=True)
    sampler.run_sampler(num_samples=500, batch_size=500)
    result = sampler.to_result()
    result.samples["chi_eff"] = inf.with_derived_parameters(result.samples, names=["chi_eff"])["chi_eff"]
    result.to_file(dst)
    with h5py.File(src, "r") as source, h5py.File(dst, "a") as target:
        source.copy("injection_parameters", target)
        if "context/signal" in source and "context/signal" not in target:
            source.copy("context/signal", target["context"])
        target.attrs["source_injection"] = src
        for group in ("waveform", "asds"):
            for det in source["context/" + group]:
                assert np.array_equal(source["context/" + group + "/" + det][()], target["context/" + group + "/" + det][()]), (group, det)
    inf._embed_model_info(dst, label, main, init)
    print("DONE", name, "seconds", time.time()-start, "identical strain and ASD verified", flush=True)
PY
cd inference_result/analysis
OUT="../figures/ET_5M_complete_best140_best10_20261021"
mkdir -p "$OUT"
SUMMARY="../figures/summary.txt"
BACKUP=$(mktemp)
HAD_SUMMARY=0
if [[ -f "$SUMMARY" ]]; then cp -p "$SUMMARY" "$BACKUP"; HAD_SUMMARY=1; fi
restore_summary() {
 if [[ "$HAD_SUMMARY" == 1 ]]; then cp -p "$BACKUP" "$SUMMARY"; else rm -f "$SUMMARY"; fi
 rm -f "$BACKUP"
}
trap restore_summary EXIT
for index in {0..9}; do
 printf -v target 'injection_%04d' "$index"
 python -u plot_posteriors.py "$target" --dingo "../posteriors/ET_5M_complete_best140/best10_20261021/$target.hdf5" --bilby "$ROOT/bilby_baseline/bilby_injections/ET_best_best10_20261008_fullspin_nlive250/$target/bilby_result.json" --outdir "$OUT" -p chirp_mass mass_ratio chi_eff luminosity_distance
 cp "$SUMMARY" "$OUT/$target/summary.txt"
 echo "PLOT DONE $target"
done
echo "All ten inference and corner plots completed $(date -Is)"
