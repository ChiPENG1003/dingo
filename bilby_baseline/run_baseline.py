#!/usr/bin/env python
import os
import sys
import argparse

# Comma- or space-separated Bilby prior names to sample, e.g.
# "chirp_mass, mass_ratio, luminosity_distance, geocent_time".
# An empty string samples all parameters in the chosen prior.
# Any omitted non-spin parameters are fixed to their injection values.
PARAMS_TO_SAMPLE = ""

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import bilby
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import (
    build_ifos,
    build_prior,
    build_waveform_generator,
    build_likelihood,
    generate_all_bbh_parameters_with_chi_eff,
)

from dingo.gw.result import Result          # noqa: E402

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

# Default to the gnpe example result, but anything can be passed with --result.
DEFAULT_RESULT = (
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/"
    "gnpe_model/inference_result/result.hdf5"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--result", default=DEFAULT_RESULT,
                    help="Path to the Dingo result.hdf5.")
    ap.add_argument("--outdir", default="bilby_run",
                    help="Output directory for the bilby run.")
    ap.add_argument("--label", default="bilby_baseline",
                    help="bilby run label (output file prefix).")
    ap.add_argument("--prior", default="LVK", choices=["LVK", "ET"],
                    help="Which prior set build_prior() uses. 'LVK' is the O2 "
                         "BBH range (chirp_mass 10-80, d_L 100-2000 Mpc); 'ET' is "
                         "the saved model metadata, including mass constraints. Picking the wrong one puts the "
                         "injected truth outside the prior support.")
    ap.add_argument("--nlive", type=int, default=250,
                    help="dynesty number of live points.")
    ap.add_argument("--npool", type=int, default=1,
                    help="Number of parallel processes for the likelihood pool.")
    ap.add_argument("--sampler", default="dynesty")
    ap.add_argument("--sample", default="acceptance-walk",
                    help="dynesty sampling method.")
    ap.add_argument("--naccept", type=int, default=20,
                    help="MCMC accepted steps per new point for dynesty.")
    ap.add_argument("--dlogz", type=float, default=0.1,
                    help="dynesty evidence stopping threshold.")
    ap.add_argument("--spin-mode", default=os.environ.get("SPIN_MODE", "full"),
                    choices=["full", "truth"],
                    help="Which spin parameters the sampler explores. 'full' "
                         "(default) leaves a_1/a_2/tilt_1/tilt_2/phi_12/phi_jl/phase "
                         "free on the training prior -- the only setting that is a "
                         "valid baseline, and the one that serves BOTH a chi_eff "
                         "network and a raw-spin network from a single run, because "
                         "generate_all_bbh_parameters derives chi_eff from the "
                         "samples. 'truth' pins those seven at the injected truth: it "
                         "hands bilby information Dingo does not have, so it is a "
                         "diagnostic/reproduction switch, NOT a baseline. Defaults "
                         "to $SPIN_MODE.")
    ap.add_argument("--free-params", default=os.environ.get("FREE_PARAMS", PARAMS_TO_SAMPLE),
                    metavar="NAMES",
                    help="Comma- or space-separated parameters to leave free; every "
                         "other sampled prior entry is pinned at its injected truth. "
                         "Overrides --spin-mode's own pinning. Example: "
                         "'chirp_mass,mass_ratio,luminosity_distance,a_1,a_2,tilt_1,"
                         "tilt_2'. Defaults to $FREE_PARAMS if set, otherwise "
                         "PARAMS_TO_SAMPLE; empty means leave the --spin-mode choice "
                         "alone. Pinning at truth hands bilby "
                         "information Dingo marginalised over, so a run like this is a "
                         "targeted study, not a baseline.")
    ap.add_argument("--check-only", action="store_true",
                    help="Run the likelihood cross-check and exit (no sampling).")
    ap.add_argument("--skip-check", action="store_true",
                    help="Skip the cross-check and go straight to sampling.")
    args = ap.parse_args()
    if args.check_only and args.skip_check:
        ap.error("--check-only and --skip-check are mutually exclusive")
    # Output directories are not HDF5 files; keep the input result suffix intact.
    args.outdir = os.path.normpath(args.outdir)
    if args.outdir.endswith(".hdf5"):
        args.outdir = args.outdir[:-len(".hdf5")]
        if not args.outdir or args.outdir.endswith(os.sep):
            ap.error("--outdir must have a directory name before .hdf5")
    return args


PINNED_BY_TRUTH = (
    "phase", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
)


def _read_truth(result_path):
    import h5py
    with h5py.File(result_path, "r") as f:
        if "injection_parameters" not in f:
            raise SystemExit(
                "Pinning parameters to truth requires /injection_parameters; "
                "use an injection result or leave PARAMS_TO_SAMPLE empty with "
                "--spin-mode full."
            )
        truth = {k: float(f[f"injection_parameters/{k}"][()])
                 for k in f["injection_parameters"]}
        if "mass_ratio" not in truth and "q" in truth:
            truth["mass_ratio"] = truth["q"]
        return truth


def apply_free_params(prior, result_path, free_params, t_ref):
    """Leave exactly `free_params` free; pin every other parameter at the truth.

    Named parameters must already exist in the prior, so a typo fails here rather
    than silently producing a run with one parameter more than intended.
    """
    free = {"mass_ratio" if key == "q" else key for key in free_params}
    unknown = sorted(free - set(prior))
    if unknown:
        raise SystemExit(
            f"PARAMS_TO_SAMPLE/--free-params names {unknown}, which the prior "
            "does not contain.\n"
            f"  available: {sorted(prior)}"
        )
    constraints = {k for k, p in prior.items()
                   if isinstance(p, bilby.core.prior.Constraint)}
    invalid = sorted(free & constraints)
    if invalid:
        raise SystemExit(f"Cannot sample derived constraints directly: {invalid}")

    to_pin = [k for k, p in prior.items()
              if k not in free and not isinstance(
                  p, (bilby.core.prior.Constraint, bilby.core.prior.DeltaFunction)
              )]
    truth = _read_truth(result_path) if to_pin else {}
    # Dingo stores geocent_time relative to t_ref while the prior is on absolute GPS.
    # Pinning the raw value would put the merger ~1.1e9 s away from the data.
    if "geocent_time" in truth:
        truth["geocent_time"] += t_ref

    missing = sorted(k for k in to_pin if k not in truth)
    if missing:
        raise SystemExit(
            f"Cannot sample exactly {sorted(free)}: injection_parameters "
            f"has no value for {missing}. Include them in PARAMS_TO_SAMPLE "
            "or use a result with those injection values."
        )

    pinned = []
    for key in list(prior):
        if key in free:
            continue
        if isinstance(prior[key], bilby.core.prior.Constraint):
            continue            # mass_1/mass_2 stay as derived constraints
        if isinstance(prior[key], bilby.core.prior.DeltaFunction):
            continue
        prior[key] = truth[key]
        pinned.append(f"{key}={truth[key]:.6g}")
    prior.convert_floats_to_delta_functions()

    print(f"parameters selected for sampling: {sorted(free)}", flush=True)
    if pinned:
        print("!! pinned at the injected truth (so this is a targeted study, "
              "not a baseline):", flush=True)
        for p in pinned:
            print(f"!!   {p}", flush=True)
    return prior


def apply_spin_mode(prior, result_path, spin_mode):

    if spin_mode == "full":
        print(
            "spin-mode full: component spins are sampled; "
            "chi_eff are derived from the samples.",
            flush=True,
        )
        return prior
    # Only remaining alternative: truth mode
    truth = _read_truth(result_path)
    pinned = []
    for key in PINNED_BY_TRUTH:
        if key in truth and key in prior:
            prior[key] = truth[key]
            pinned.append(f"{key}={truth[key]:.6g}")
    prior.convert_floats_to_delta_functions()
    print(
        "!! spin-mode truth: NOT a valid baseline -- "
        "spin parameters are pinned at injected truth.",
        flush=True,
    )
    for p in pinned:
        print(f"!!   {p}", flush=True)

    return prior


def check_injection_likelihood(result, result_path, ifos, wfg, likelihood):
    """Compare Bilby's waveform and log likelihood ratio to the saved signal."""
    import h5py
    import numpy as np

    with h5py.File(result_path, "r") as f:
        if "context/signal" not in f or "injection_parameters" not in f:
            raise ValueError("Injection check needs saved signal and truth; "
                             "use --skip-check explicitly for other data.")
        signal = {key: np.asarray(value) for key, value in f["context/signal"].items()}
    truth = _read_truth(result_path)
    # Use original waveform parameters, never substitute a derived spin.
    for key in ("chi_eff", "chi_p", "chi_1z", "chi_2z"):
        truth.pop(key, None)
    truth["geocent_time"] += result.t_ref
    polarizations = wfg.frequency_domain_strain(truth)
    residual = 0.0
    expected = 0.0
    for ifo in ifos:
        mask = ifo.frequency_mask
        h = signal[ifo.name][mask]
        model = ifo.get_detector_response(polarizations, truth)[mask]
        data = ifo.frequency_domain_strain[mask]
        psd = ifo.power_spectral_density_array[mask]
        factor = 4.0 / ifo.duration
        residual += factor * np.sum(np.abs(model - h) ** 2 / psd)
        expected += factor * np.sum((np.real(np.conj(h) * data)
                                    - 0.5 * np.abs(h) ** 2) / psd)
    actual = likelihood.log_likelihood_ratio(parameters=truth)
    if not (np.isfinite(residual) and np.isfinite(actual)
            and residual < 1e-2 and abs(actual - expected) < 0.1):
        raise ValueError(f"Injection cross-check failed: residual={residual}, "
                         f"logLR={actual}, saved-signal logLR={expected}")
    print(f"Injection cross-check OK: residual={residual:.6g}, "
          f"logLR difference={actual - expected:.6g}", flush=True)


def main():
    args = parse_args()
    free_params = args.free_params.replace(",", " ").split()
    if "chi_eff" in free_params:
        raise SystemExit("chi_eff is derived from component-spin posterior samples; "
                         "leave --free-params empty for the full baseline.")

    print(f"Loading Dingo result: {args.result}", flush=True)
    result = Result(file_name=args.result)
    print("Building bilby data/prior/waveform", flush=True)
    ifos, duration, sampling_frequency = build_ifos(result)
    prior = build_prior(result, prior=args.prior)
    if free_params:
        prior = apply_free_params(prior, args.result, free_params, result.t_ref)
    else:
        prior = apply_spin_mode(prior, args.result, args.spin_mode)
    wfg = build_waveform_generator(
        result,
        duration,
        sampling_frequency
    )

    free = sorted(k for k, p in prior.items()
                  if not isinstance(p, bilby.core.prior.DeltaFunction)
                  and not isinstance(p, bilby.core.prior.Constraint))
    print(f"sampling {len(free)} parameters: {free}", flush=True)

    # Plain (non-marginalized) likelihood for both the Dingo cross-check and
    # sampling (cf. tutorial Lab 3).
    likelihood = build_likelihood(ifos, wfg)

    if not args.skip_check:
        check_injection_likelihood(result, args.result, ifos, wfg, likelihood)
    if args.check_only:
        print("Check-only completed; no sampler started.", flush=True)
        return

    print("Starting sampler", flush=True)
    bilby.run_sampler(
        likelihood=likelihood,
        priors=prior,
        sampler=args.sampler,
        nlive=args.nlive,
        sample=args.sample,
        naccept=args.naccept,
        dlogz=args.dlogz,
        npool=args.npool,
        outdir=args.outdir,
        label=args.label,
        check_point_delta_t=50,
        conversion_function=generate_all_bbh_parameters_with_chi_eff,
        result_class=bilby.gw.result.CompactBinaryCoalescenceResult,
    )


if __name__ == "__main__":
    main()
