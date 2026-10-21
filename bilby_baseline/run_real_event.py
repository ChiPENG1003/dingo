#!/usr/bin/env python
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import sys

import bilby
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import (
    build_ifos,
    build_prior,
    build_waveform_generator,
    build_likelihood,
)

from dingo.gw.domains import build_domain

try:
    import torch
    torch.set_num_threads(1)
except Exception:
    pass

REAL_EVENT_DIR = (
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/examples/"
    "gnpe_model/inference/inference_result/events"
)
DEFAULT_OUTDIR = (
    "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline/bilby_real_events"
)

DEFAULT_APPROXIMANT = "IMRPhenomPv2"
DEFAULT_F_REF = 20.0
DEFAULT_SPIN_CONVERSION_PHASE = 0.0


def _scalar(dataset):
    value = dataset[()]
    return value.decode() if isinstance(value, bytes) else value


class RealEventResult:
    def __init__(self, path, approximant=DEFAULT_APPROXIMANT, f_ref=DEFAULT_F_REF,
                 spin_conversion_phase=DEFAULT_SPIN_CONVERSION_PHASE):
        with h5py.File(path, "r") as f:
            if "context" not in f or "event_metadata" not in f:
                raise ValueError(
                    f"{path} is not a real-event data file (expected 'context' and "
                    f"'event_metadata'; found {list(f.keys())})."
                )
            meta = {k: _scalar(f["event_metadata"][k])
                    for k in f["event_metadata"].keys()
                    if f["event_metadata"][k].shape == ()}
            detectors = [d.decode() if isinstance(d, bytes) else str(d)
                         for d in f["event_metadata"]["detectors"][()]]
            self.context = {
                "waveform": {d: np.asarray(f["context"]["waveform"][d][()],
                                           dtype=complex) for d in detectors},
                "asds": {d: np.asarray(f["context"]["asds"][d][()], dtype=float)
                         for d in detectors},
            }

        self.event_metadata = meta
        self.interferometers = detectors

        self.t_ref = float(meta["time_event"])

        duration = float(meta["T"])
        self.domain = build_domain({
            "type": "UniformFrequencyDomain",
            "f_min": float(meta["minimum_frequency"]),
            "f_max": float(meta["maximum_frequency"]),
            "delta_f": 1.0 / duration,
        })

        n_expected = len(self.domain.sample_frequencies)
        for det, strain in self.context["waveform"].items():
            if len(strain) != n_expected:
                raise ValueError(
                    f"[{det}] strain has {len(strain)} bins but the domain "
                    f"(f_max={meta['maximum_frequency']}, T={duration}) implies "
                    f"{n_expected}."
                )

        self.base_metadata = {
            "dataset_settings": {
                "waveform_generator": {
                    "approximant": approximant,
                    "f_ref": f_ref,
                    "spin_conversion_phase": spin_conversion_phase,
                }
            }
        }

    @property
    def event_name(self):
        return self.event_metadata.get("event_name", "real_event")


def run_one(event_file, outdir, nlive, npool, sampler="dynesty",
            sample="acceptance-walk", naccept=20, dlogz=0.1,
            approximant=DEFAULT_APPROXIMANT, f_ref=DEFAULT_F_REF,
            spin_conversion_phase=DEFAULT_SPIN_CONVERSION_PHASE):
    label = "bilby"
    # bilby writes JSON by default, so checking for .hdf5 here never matched and
    # the "already done" guard silently did nothing.
    result_file = os.path.join(outdir, f"{label}_result.json")
    if os.path.exists(result_file):
        print(f"already done -> {result_file}, skipping.")
        return result_file

    # dynesty resumes from <outdir>/<label>_resume.pickle whenever one exists, and
    # the restored sampler keeps ITS nlive/naccept -- the values passed here are
    # ignored. A leftover checkpoint from a different configuration therefore
    # produces a run that looks new but reproduces the old settings.
    resume_file = os.path.join(outdir, f"{label}_resume.pickle")
    if os.path.exists(resume_file):
        print(f"WARNING: resuming from {resume_file}. nlive={nlive} and "
              f"naccept={naccept} will be IGNORED in favour of whatever the "
              f"checkpoint was created with. Move or delete {outdir} to start "
              f"a genuinely fresh run.", flush=True)

    os.makedirs(outdir, exist_ok=True)

    print(f"loading {event_file}", flush=True)
    result = RealEventResult(
        event_file, approximant=approximant, f_ref=f_ref,
        spin_conversion_phase=spin_conversion_phase,
    )
    print(f"event {result.event_name}  trigger GPS {result.t_ref}  "
          f"detectors {result.interferometers}  "
          f"f in [{result.domain.f_min}, {result.domain.f_max}] Hz", flush=True)

    print("building bilby data/prior/waveform", flush=True)
    ifos, duration, sampling_frequency = build_ifos(result)
    prior = build_prior(result)
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
        outdir=outdir,
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
    ap.add_argument("--event", default="GW170809",
                    help="Event name; resolves to <real_event_dir>/<event>.hdf5.")
    ap.add_argument("--event_file", default=None,
                    help="Explicit real-event HDF5, overriding --event.")
    ap.add_argument("--real_event_dir", default=REAL_EVENT_DIR)
    ap.add_argument("--outdir", default=None,
                    help=f"Default: {DEFAULT_OUTDIR}/<event>")
    ap.add_argument("--nlive", type=int, default=250,
                    help="dynesty live points (sampler resolution, NOT the "
                         "number of posterior samples).")
    ap.add_argument("--npool", type=int, default=1)
    ap.add_argument("--sampler", default="dynesty")
    ap.add_argument("--sample", default="acceptance-walk")
    ap.add_argument("--naccept", type=int, default=20)
    ap.add_argument("--dlogz", type=float, default=0.1)
    ap.add_argument("--approximant", default=DEFAULT_APPROXIMANT)
    ap.add_argument("--f_ref", type=float, default=DEFAULT_F_REF)
    ap.add_argument("--spin_conversion_phase", type=float,
                    default=DEFAULT_SPIN_CONVERSION_PHASE,
                    help="Match Dingo's convention (0.0) so the baseline is "
                         "comparable to the Dingo posterior.")
    args = ap.parse_args()

    event_file = args.event_file or os.path.join(
        args.real_event_dir, f"{args.event}.hdf5"
    )
    if not os.path.exists(event_file):
        raise FileNotFoundError(
            f"{event_file} not found. Generate it first:\n"
            f"  python examples/gnpe_model/construct_real_event.py --event {args.event}"
        )
    outdir = args.outdir or os.path.join(DEFAULT_OUTDIR, args.event)

    run_one(event_file, outdir, args.nlive, args.npool,
            sampler=args.sampler, sample=args.sample, naccept=args.naccept,
            dlogz=args.dlogz, approximant=args.approximant, f_ref=args.f_ref,
            spin_conversion_phase=args.spin_conversion_phase)


if __name__ == "__main__":
    main()
