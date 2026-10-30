#!/usr/bin/env python
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import argparse
import bilby
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import build_ifos, build_prior, build_waveform_generator, build_likelihood

from dingo.gw.result import Result
try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# inference.py's output tree, as it stands after the inference/ refactor.
INFERENCE_DIR = (
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/"
    "gnpe_model/inference/inference_result"
)
DEFAULT_INJ_DIR = os.path.join(INFERENCE_DIR, "posteriors/1M_7params_O2/series")
DEFAULT_OUTDIR = (
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/bilby_injections"
)

from dingo.gw.injection import Injection          # noqa: E402

SAMPLED_PARAMETERS = (
    "chirp_mass", "mass_ratio", "luminosity_distance",
    "chi_eff", "ra", "dec", "psi", "geocent_time",
)


def injection_path(inj_dir,inj_name):
    path = os.path.join(inj_dir, f"{inj_name}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{inj_name} not found in {inj_dir}")
    return path


def load_truth(path, result):
    with h5py.File(path, "r") as f:
        if "injection_parameters" in f:
            g = f["injection_parameters"]
            return {k: float(g[k][()]) for k in g.keys()}
    # Older files carry no embedded truth and it has to be redrawn from the seed.
    # Imported here rather than at module scope so that the normal path does not
    # depend on the analysis package's location.
    sys.path.insert(0, os.path.join(INFERENCE_DIR, "analysis"))
    import coverage_stats as cs  # type: ignore

    dingo_prior = Injection.from_posterior_model_metadata(result.metadata).prior
    seed = (result.event_metadata or {}).get("injection_seed")
    return cs.reconstruct_truth(dingo_prior, seed)


def fix_non_sampled_to_truth(prior, result, sampled, path):
    truth = load_truth(path, result)                     # relative geocent_time
    for key in list(prior.keys()):
        p = prior[key]
        if not isinstance(p, bilby.core.prior.Prior):
            continue
        if isinstance(p, bilby.core.prior.Constraint):   # keep mass_1/mass_2
            continue
        if key in sampled or key not in truth:
            continue
        value = truth[key]
        if key == "geocent_time":                        # truth is relative
            value = result.t_ref + value
        prior[key] = float(value)                        # assigning a float fixes it
    return prior


def validate_prior(prior, result, path):
    truth = load_truth(path, result)
    truth["geocent_time"] += result.t_ref
    converted = prior.conversion_function(dict(truth))
    for key, p in prior.items():
        if isinstance(p, bilby.core.prior.Prior):
            if key not in converted or not np.isfinite(p.ln_prob(converted[key])):
                raise ValueError(f"Injected truth outside prior: {key}={converted.get(key)}, {p}")
    print("preflight OK: injected truth satisfies all priors and mass constraints", flush=True)


def run_one(inj_dir, inj_name, outdir, nlive, npool, prior, sampler="dynesty",
            sample="acceptance-walk", naccept=20, dlogz=1.0,
            sampled_parameters=SAMPLED_PARAMETERS):
    label = "bilby"
    run_dir = outdir
    result_file = os.path.join(run_dir, f"{label}_result.hdf5")
    if os.path.exists(result_file):
        print(f"already done -> {result_file}, skipping.")
        return result_file

    os.makedirs(outdir, exist_ok=True)

    path = injection_path(inj_dir,inj_name)
    print(f"loading {path}", flush=True)
    result = Result(file_name=path)

    print("building bilby data/prior/waveform", flush=True)
    print(f"prior set: {prior}", flush=True)
    prior = build_prior(result, prior = prior)
    validate_prior(prior, result, path)
    prior = fix_non_sampled_to_truth(prior, result, sampled_parameters, path)
    print(f"effective priors: {prior}", flush=True)
    ifos, duration, sampling_frequency = build_ifos(result)
    wfg = build_waveform_generator(result, duration, sampling_frequency)
    likelihood = build_likelihood(ifos, wfg)
    sampling = sorted(k for k, v in prior.items()
                      if isinstance(v, bilby.core.prior.Prior)
                      and not isinstance(v, bilby.core.prior.Constraint))
    print(f"sampling {len(sampling)} params: {sampling}", flush=True)

    bilby.run_sampler(
        likelihood=likelihood,
        priors=prior,
        sampler=sampler,
        nlive=nlive,
        sample=sample,
        naccept=naccept,
        dlogz=dlogz,
        npool=npool,
        outdir=run_dir,
        label=label,
        check_point_delta_t=100,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CompactBinaryCoalescenceResult,
    )
    print(f"done -> {result_file}", flush=True)
    return result_file


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inj_dir", default=DEFAULT_INJ_DIR)
    # $inference_type names the run's output folder; without it the folder is
    # named after the injection file, so the default is always a valid path.
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--nlive", type=int, default=250,
                    help="dynesty live points (sampler resolution, NOT the "
                         "number of posterior samples).")
    ap.add_argument("--inj_name", default="injection.hdf5")
    ap.add_argument("--npool", type=int, default=1,
                    help="Processes for the sampler. Keep 1 for a robust serial "
                         "run; >1 uses a multiprocessing pool.")
    ap.add_argument("--sample", default="acceptance-walk",
                    help="dynesty sampling method.")
    ap.add_argument("--naccept", type=int, default=20,
                    help="MCMC accepted steps per new point (lower = faster).")
    ap.add_argument("--dlogz", type=float, default=0.1,
                    help="Evidence stopping threshold (higher = faster).")
    ap.add_argument("--sampler", default="dynesty")
    ap.add_argument("--prior", default="LVK", choices=["LVK", "ET"])
    ap.add_argument("--sample-params", default=None,
                    help="Comma-separated parameters to sample (Level 2). Others "
                         "are fixed to the injected truth. Default: "
                         + ",".join(SAMPLED_PARAMETERS))
    args = ap.parse_args()

    sampled = (tuple(s.strip() for s in args.sample_params.split(","))
               if args.sample_params else SAMPLED_PARAMETERS)

    outdir = args.outdir or os.path.join(
        DEFAULT_OUTDIR,
        os.environ.get("inference_type") or os.path.splitext(args.inj_name)[0])

    print(f"Sampling parameters: {sampled}", flush=True)
    print(f"outdir: {outdir}", flush=True)
    run_one(args.inj_dir, args.inj_name, outdir, args.nlive, args.npool, args.prior,
            sampler=args.sampler, sample=args.sample,
            naccept=args.naccept, dlogz=args.dlogz,
            sampled_parameters=sampled)


if __name__ == "__main__":
    main()
