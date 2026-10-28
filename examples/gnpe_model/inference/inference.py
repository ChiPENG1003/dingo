import argparse
import copy
import json
import os
import sys
import time

import bilby
import h5py
import numpy as np
import torch

from dingo.core.posterior_models.build_model import build_model_from_kwargs
import dingo.gw.injection as injection
from dingo.gw.derived_parameters import with_derived_parameters
from dingo.gw.domains import build_domain_from_model_metadata
from dingo.gw.gwutils import get_extrinsic_prior_dict
from dingo.gw.inference.gw_samplers import GWSamplerGNPE, GWSampler
from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.prior import build_prior_with_defaults
from dingo.gw.transforms import GetDetectorTimes, PostCorrectGeocentTime
from dingo.gw.transforms.overlap_transforms import finite_window_time_shift
from torchvision.transforms import Compose

INFERENCE_DIR = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(INFERENCE_DIR)

TRAINING_DIR = os.path.join(EXAMPLE_DIR, "training")
CONSTRUCT_DIR = os.path.join(EXAMPLE_DIR, "construct_plot_data")

RESULT_ROOT = os.path.join(INFERENCE_DIR, "inference_result")
REAL_EVENT_INPUT_DIR = os.path.join(RESULT_ROOT, "events")
POSTERIOR_DIR = os.path.join(RESULT_ROOT, "posteriors")


DEFAULT_MODEL = "ET_200"

MODEL_REGISTRY = {
    # "1M200SVD": (
    #     "main_train_dir_1M_2000train_500val_200SVD/best_model.pt",
    #     "init_train_dir_1M_20000train_5000val_200SVD/model_latest.pt",
    # ),
    "1M200SVD_7params": (
        "main_train_dir_1M_20000train_5000val_200SVD_7params/model_400.pt",
        "init_train_dir_7params_O2/model_latest.pt",
    ),
    "1M_7params_O2": (
        "main_train_dir_1M_7params_O2/model_400.pt",
        "init_train_dir_7params_O2/model_latest.pt",
    ),
    "chi_eff": (
        "main_train_dir_1M_chi_eff/model_latest.pt",
        "init_train_dir_7params_O2/model_latest.pt",
    ),
    "ET_150": (
        "main_train_dir_ET/model_150.pt",
        "init_train_dir_ET/model_latest.pt",
    ),
    "ET_org": (
        "main_train_dir_ET/model_150.pt",
        "init_train_dir_ET/model_latest.pt",
    ),
    "ET_chi": (
        "main_train_dir_ET_5M_chi/model_latest.pt",
        "init_train_dir_ET/model_latest.pt",
    ),
}

LEGACY_MODE_ENV = {
    "single": "single",
    "overlap": "overlap",
    "multi": "multi",
    "overlap_multi": "overlap-multi",
    "real": "real",
    "compare": "compare",
}


def _env(name, cast, default):
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        raise SystemExit(f"Environment variable {name}={raw!r} is not a valid {cast.__name__}")


def _add_common(parser):
    parser.add_argument("--num-samples", type=int, default=_env("NUM_SAMPLES", int, 5000),
                        help="Posterior samples to draw.")
    parser.add_argument("--batch-size", type=int, default=_env("BATCH_SIZE", int, 5000))
    parser.add_argument("--num-iterations", type=int, default=_env("NUM_ITERATIONS", int, 30),
                        help="GNPE Gibbs iterations.")
    parser.add_argument("--allow-incomplete-gnpe", action="store_true",
                        help="Allow approximate, single-pass inference when the main "
                             "model omits ra, dec, or geocent_time. Not a converged "
                             "GNPE posterior; the limitation is saved in metadata.")


def _add_model(parser):
    parser.add_argument("--model", default=_env("MODEL", str, DEFAULT_MODEL),
                        metavar="MODEL",
                        help=f"Which trained network to use: a MODEL_REGISTRY key "
                             f"({', '.join(MODEL_REGISTRY)}), or an explicit "
                             f"'[label=]main.pt|init.pt' with relative paths resolving "
                             f"under training/. Defaults to $MODEL, else DEFAULT_MODEL "
                             f"= {DEFAULT_MODEL!r} at the top of inference.py. "
                             f"'inference.py models' lists them.")


def _add_injection(parser):
    """Knobs that change the injected data itself, and so the output filename."""
    parser.add_argument("--distance", type=float, default=_env("FIX_DISTANCE", float, 80.0),
                        metavar="MPC",
                        help="Fix luminosity_distance to a draw from N(distance, distance-std). "
                             "0 keeps the model's own prior, the only setting guaranteed to "
                             "stay inside the trained support.")
    parser.add_argument("--distance-std", type=float,
                        default=_env("FIX_DISTANCE_STD", float, 25.0), metavar="MPC")


def _add_overlap(parser):
    parser.add_argument("--shift", type=float, default=_env("SHIFT_SECONDS", float, 0.5),
                        metavar="SECONDS",
                        help="How much the second signal is moved in time. "
                             "_linear_time_shift moves it EARLIER for a positive "
                             "value (combo_common.verify_shift_direction measures it).")
    parser.add_argument("--second-seed-offset", type=int,
                        default=_env("SECOND_SEED_OFFSET", int, 1000000),
                        help="Seed of the second signal = seed + this.")
    # Controlled parameters of the distraction signal. Everything not named here
    # stays as the prior drew it, so a sweep over one of these varies exactly one
    # thing. 0 means 'leave the prior draw alone' -- neither is a physical value.
    parser.add_argument("--second-chirp-mass", type=float,
                        default=_env("SECOND_CHIRP_MASS", float, 0.0), metavar="MSUN",
                        help="Pin the second signal's chirp mass. The mass ratio "
                             "stays as drawn, so the component masses follow; they "
                             "may leave the prior's Constraint(10, 80) range, which "
                             "is allowed for an injected signal nobody models.")
    parser.add_argument("--second-distance", type=float,
                        default=_env("SECOND_DISTANCE", float, 0.0), metavar="MPC",
                        help="Pin the second signal's luminosity distance, exactly "
                             "(unlike --distance, which draws around a mean).")
    parser.add_argument("--free-second-time", action="store_true",
                        default=_env("FREE_SECOND_TIME", int, 0) == 1,
                        help="Let the second signal keep its own geocent_time draw. "
                             "By default it is tied to signal 1's, so --shift IS the "
                             "merger separation; the prior is Uniform(-0.1, 0.1) s, "
                             "which would otherwise smear the separation by +-0.2 s.")


def build_parser():
    parser = argparse.ArgumentParser(
        # A literal, not __doc__: the parser must keep working if the module
        # docstring is edited away, and `python -OO` strips docstrings entirely.
        description="Posterior sampling for the GNPE models trained in this directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", metavar="MODE")

    tag_default = os.environ.get("SINGLE_OUT_NAME", "")
    tag_default = "" if tag_default == "injection" else tag_default

    def _one_injection(name, help_text):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        _add_model(p)
        _add_injection(p)
        p.add_argument("--seed", type=int, default=_env("SINGLE_SEED", int, 0),
                       help="Seeds the truth draw AND the noise realization.")
        p.add_argument("--tag", default=tag_default,
                       help="Extra suffix on the output filename.")
        return p

    _one_injection("single", "One injection, one model.")
    _add_overlap(_one_injection("overlap", "One injection plus a time-shifted second signal."))

    def _many_injections(name, help_text):
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
        _add_model(p)
        p.add_argument("--num-injections", type=int,
                       default=_env("NUM_INJECTIONS", int, 1000))
        p.add_argument("--base-seed", type=int, default=_env("BASE_SEED", int, 0),
                       help="Injection i uses seed base-seed + i.")
        p.add_argument("--max-retries", type=int, default=_env("MAX_RETRIES", int, 3),
                       help="Times to re-draw an injection whose flow sampling "
                            "raises before skipping it.")
        p.add_argument("--max-consecutive-failures", type=int,
                       default=_env("MAX_CONSECUTIVE_FAILURES", int, 5),
                       help="Abort once this many injections fail back to back. "
                            "Isolated failures are bad draws; a run of them is a "
                            "broken environment, and skipping 500 of those in a "
                            "row would otherwise 'succeed' with no output.")
        # The two controlled variants: hold one of (truth, noise) fixed and vary
        # the other, instead of varying both as a plain population run does.
        p.add_argument("--fixed-truth", default=_env("FIXED_TRUTH", str, ""),
                       help="Path to a result HDF5 whose /injection_parameters are "
                            "reused for EVERY injection. Only the noise realization "
                            "then varies (seed base-seed + i).")
        p.add_argument("--fixed-noise-seed", type=int,
                       default=_env("FIXED_NOISE_SEED", int, -1),
                       help="Draw the SAME noise realization for every injection. "
                            "Only the truth then varies. -1 (default) = fresh noise.")
        p.add_argument("--out-name", default=_env("OUT_NAME", str, ""),
                       help="Output subdirectory under posteriors/<model>/ instead "
                            "of '<mode>'. Keeps a controlled experiment beside, not "
                            "on top of, an existing population run.")
        return p

    _many_injections("multi", "Many injections, for population and coverage studies.")
    _add_overlap(_many_injections("overlap-multi", "Many two-signal injections."))

    p_real = sub.add_parser("real", help="Observed strain from construct_real_event.py.")
    _add_common(p_real)
    _add_model(p_real)
    p_real.add_argument("--event-file",
                        default=_env("REAL_EVENT_FILE", str,
                                     os.path.join(REAL_EVENT_INPUT_DIR, "GW150914.hdf5")),
                        help="Context file written by construct_real_event.py.")

    p_cmp = sub.add_parser("compare", help="Several models on ONE injection.")
    _add_common(p_cmp)
    _add_injection(p_cmp)
    p_cmp.add_argument("--models", default=_env("COMPARE_MODELS", str, "300SVD,200SVD,50k300SVD"),
                       help="Comma-separated. The FIRST entry is the reference: it fixes "
                            "the truth, the prior and the noise draw that every other "
                            "model is then handed unchanged.")
    p_cmp.add_argument("--seed", type=int, default=_env("SINGLE_SEED", int, 0))
    p_cmp.add_argument("--tag", default=tag_default)

    sub.add_parser("models", help="List the model registry and check the checkpoints exist.")

    return parser


def parse_args(argv=None):
    """Parse argv, falling back to the legacy MODE environment variable."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    if not argv or argv[0].startswith("-"):
        legacy = os.environ.get("MODE")
        if legacy is None:
            parser.print_help()
            raise SystemExit(1)
        if legacy not in LEGACY_MODE_ENV:
            raise SystemExit(
                f"Unknown MODE={legacy!r}. Known: {sorted(LEGACY_MODE_ENV)}"
            )
        mode = LEGACY_MODE_ENV[legacy]
        print(f"(MODE={legacy} from the environment -> subcommand '{mode}'; "
              f"'python inference.py {mode} --help' lists its options)")
        argv = [mode] + argv

    return parser.parse_args(argv)


# --------------------------------------------------------------------------
# Output layout -- the single place that decides where anything is written
# --------------------------------------------------------------------------

def _seed_tag(seed):
    return f"seed{int(seed):04d}"


def _distance_tag(distance):
    """0 means 'drawn from the model prior', which is not a number to put in a name."""
    return "dprior" if not distance else f"d{int(round(float(distance))):04d}mpc"


def _shift_tag(shift_seconds):
    return f"shift{float(shift_seconds):.3f}s"


def _second_tags(args):
    """Name parts for whatever was pinned on the second signal.

    Without these, two overlaps that differ only in the distraction signal would
    land on the same filename and the second run would overwrite the first.
    """
    parts = []
    if getattr(args, "second_chirp_mass", 0):
        parts.append(f"mc2_{float(args.second_chirp_mass):g}")
    if getattr(args, "second_distance", 0):
        parts.append(f"d2_{float(args.second_distance):g}mpc")
    if getattr(args, "free_second_time", False):
        parts.append("freet2")
    return parts


def _stem(seed, distance, shift_seconds=None, tag="", extra=()):
    parts = [_seed_tag(seed), _distance_tag(distance)]
    if shift_seconds is not None:
        parts.append(_shift_tag(shift_seconds))
    parts.extend(extra)
    if tag:
        parts.append(tag)
    return "_".join(parts)


def mode_dir(mode, model_label, shift_seconds=None):
    """Directory for one mode's results. Model name is part of the path so that
    re-running with a different network cannot overwrite an earlier one."""
    parts = [POSTERIOR_DIR, model_label, mode.replace("-", "_")]
    if shift_seconds is not None:
        parts.append(_shift_tag(shift_seconds))
    return os.path.join(*parts)


def _rel(path):
    return os.path.relpath(path, EXAMPLE_DIR)


def _announce(mode, model_label, inputs, out_dir, outputs):
    width = 78
    print("=" * width)
    print(f" MODE={mode}   model={model_label}")
    print("-" * width)
    print(" inputs")
    for name, value in inputs.items():
        print(f"   {name:16s} {value}")
    print(f" outputs -> {_rel(out_dir)}/")
    for name in outputs:
        print(f"   {name}")
    print("=" * width, flush=True)


# --------------------------------------------------------------------------
# Model / sampler construction
# --------------------------------------------------------------------------

def _parse_model_specs(spec):
    """Parse a model spec list into [(label, main_path, init_path), ...].

    Entries are either a MODEL_REGISTRY key ("300SVD") or an explicit
    "label=main.pt|init.pt" pair. Relative paths resolve under training/.
    """
    def _resolve(p):
        return p if os.path.isabs(p) else os.path.join(TRAINING_DIR, p)

    specs = []
    for entry in (e.strip() for e in spec.split(",") if e.strip()):
        if "=" in entry:
            label, paths = entry.split("=", 1)
            if "|" not in paths:
                raise ValueError(f"Model spec {entry!r} must be 'label=main.pt|init.pt'.")
            main_path, init_path = paths.split("|", 1)
        elif "|" in entry:
            # Explicit pair with no label: name it after the training directory,
            # which is what makes the output path identifiable.
            main_path, init_path = entry.split("|", 1)
            label = os.path.basename(os.path.dirname(main_path.strip())) or "model"
            label = label.replace("main_train_dir_", "")
        elif entry in MODEL_REGISTRY:
            label = entry
            main_path, init_path = MODEL_REGISTRY[entry]
        else:
            raise ValueError(
                f"Unknown model {entry!r}. Known: {sorted(MODEL_REGISTRY)}, "
                f"or use 'label=main.pt|init.pt'."
            )
        specs.append((label.strip(), _resolve(main_path.strip()), _resolve(init_path.strip())))
    if not specs:
        raise ValueError("No models given.")
    return specs


def _load_pair(main_path, init_path):
    for p in (main_path, init_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    main_pm = build_model_from_kwargs(filename=main_path, device="cuda",
                                      load_training_info=False)
    init_pm = build_model_from_kwargs(filename=init_path, device="cuda",
                                      load_training_info=False)
    return main_pm, init_pm


def checkpoint_summary(main_path):
    """What a checkpoint was trained on, read without loading its weights.

    Every fact here already lives in the .pt, so it is never written down a
    second time in MODEL_REGISTRY -- a hand-copied parameter list would drift
    from the checkpoint the moment a model is retrained.

    torch.load(mmap=True) maps the tensors lazily instead of reading them, so
    this costs about a second on a 1.5 GB checkpoint rather than a full load.
    """
    import torch  # local: keep `models` usable even in a torch-less environment

    try:
        ckpt = torch.load(main_path, map_location="cpu", mmap=True,
                          weights_only=False)
    except Exception as e:  # a truncated or half-written checkpoint
        return {"error": f"{type(e).__name__}: {e}"}

    md = ckpt["metadata"]
    data = md["train_settings"]["data"]
    domain = md["dataset_settings"]["domain"]
    stages = md["train_settings"]["training"]
    last_stage = max((k for k in stages if k.startswith("stage_")),
                     key=lambda k: int(k.split("_")[1]))
    params = list(data["inference_parameters"])
    return {
        "inference_parameters": params,
        # The GNPE Gibbs loop needs sky position and geocentric time;
        # see build_sampler(). Worth surfacing when picking a model.
        "gnpe_full_loop": {"ra", "dec", "geocent_time"}.issubset(params),
        "detectors": list(data["detectors"]),
        "f_min": domain["f_min"],
        "f_max": domain["f_max"],
        "delta_f": domain["delta_f"],
        "last_stage": last_stage,
        "asd_dataset_path": stages[last_stage]["asd_dataset_path"],
        "epoch": ckpt.get("epoch", "?"),
    }


def _loaded_model_params(main_pm):
    """Line for the run banner: what this network actually regresses.

    Read from the loaded model, not from MODEL_REGISTRY, so it cannot disagree
    with the checkpoint being used.
    """
    params = main_pm.metadata["train_settings"]["data"]["inference_parameters"]
    missing = sorted({"ra", "dec", "geocent_time"} - set(params))
    note = "" if not missing else f"   !! missing {', '.join(missing)} -> GNPE cannot iterate"
    return f"{len(params)}: {', '.join(params)}{note}"


def build_models(model):
    """Load the (main, init) GNPE pair named by `model`. Returns (label, main, init)."""
    label, main_path, init_path = _parse_model_specs(model)[0]
    return label, main_path, init_path


def model_asd_path(main_pm):
    stages = main_pm.metadata["train_settings"]["training"]
    last_stage = max((k for k in stages if k.startswith("stage_")),
                     key=lambda k: int(k.split("_")[1]))
    return os.path.join(EXAMPLE_DIR, stages[last_stage]["asd_dataset_path"]), last_stage


def build_injection_generator(main_pm):
    injection_generator = injection.Injection.from_posterior_model_metadata(main_pm.metadata)
    asd_fname, _ = model_asd_path(main_pm)
    asd_dataset = ASDDataset(file_name=asd_fname)
    injection_generator.asd = {
        detector: asds[0] for detector, asds in asd_dataset.asds.items()
    }
    return injection_generator


def build_sampler(main_pm, init_pm, num_iterations=30, allow_incomplete_gnpe=False):
    inference_parameters = main_pm.metadata["train_settings"]["data"]["inference_parameters"]
    missing = sorted({"ra", "dec", "geocent_time"} - set(inference_parameters))
    requested_iterations = num_iterations
    if missing:
        reason = (
            f"Main model does not infer {', '.join(missing)}. The GNPE detector-time "
            "update requires ra, dec, and geocent_time, so this checkpoint cannot "
            "run the full Gibbs loop. Use a checkpoint trained with these targets "
            "or explicitly pass --allow-incomplete-gnpe for an APPROXIMATE single pass."
        )
        if not allow_incomplete_gnpe:
            raise ValueError(reason)
        print(f"WARNING: {reason} Forcing num_iterations=1 (requested "
              f"{requested_iterations}). Proxies come from the init network only; "
              "this is not a converged GNPE posterior.", flush=True)
        num_iterations = 1

    init_sampler = GWSampler(model=init_pm)
    sampler = GWSamplerGNPE(model=main_pm, init_sampler=init_sampler,
                            num_iterations=num_iterations)
    if missing:
        # No next iteration needs detector times. Preserve the time correction
        # whenever geocent_time is predicted: dropping it would shift the posterior.
        sampler.transform_post = Compose([
            t for t in sampler.transform_post.transforms
            if not isinstance(t, GetDetectorTimes)
            and not ("geocent_time" in missing and isinstance(t, PostCorrectGeocentTime))
        ])
        sampler.metadata["incomplete_gnpe"] = {
            "approximate": True,
            "missing_parameters": missing,
            "requested_iterations": requested_iterations,
            "actual_iterations": 1,
            "description": "Single pass using init-network proxies; not converged GNPE.",
        }
    return sampler


# --------------------------------------------------------------------------
# Injection construction and sampling
# --------------------------------------------------------------------------

def _draw_reproducible_truth(injection_generator, seed):
    bilby.core.utils.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return {name: float(value) for name, value in injection_generator.prior.sample().items()}


def _draw_truth(injection_generator, seed, distance, distance_std):

        theta = _draw_reproducible_truth(injection_generator, seed)
        if distance:
            theta["luminosity_distance"] = np.random.normal(loc=distance, scale=distance_std)
        return theta


def _apply_second_overrides(theta2, theta1, args):
    notes = []
    if not args.free_second_time:
        theta2["geocent_time"] = float(theta1["geocent_time"])
        notes.append(f"geocent_time tied to signal 1 ({theta2['geocent_time']:+.6f} s), "
                     f"so the separation is the applied shift")
    if args.second_chirp_mass:
        theta2["chirp_mass"] = float(args.second_chirp_mass)
        notes.append(f"chirp_mass pinned to {theta2['chirp_mass']:g} Msun")
    if args.second_distance:
        theta2["luminosity_distance"] = float(args.second_distance)
        notes.append(f"luminosity_distance pinned to {theta2['luminosity_distance']:g} Mpc")
    return theta2, notes


def _print_truth(theta, header):
    print(header)
    for name, value in with_derived_parameters(theta).items():
        print(f"  {name:20s} = {value: .6f}")


def _linear_time_shift(domain, h_freq, shift_seconds):
    return finite_window_time_shift(domain, h_freq, shift_seconds)


def _build_injection(injection_generator, theta, theta2=None, shift_seconds=None):
    domain = injection_generator.data_domain

    signal1 = injection_generator.signal(theta)
    try:
        asd = signal1["asds"]
    except KeyError:
        raise ValueError("self.asd must be set in order to produce injections.")

    if injection_generator.whiten:
        print("injection_generator.whiten was True. Resetting to False.")
        injection_generator.whiten = False

    signal2 = injection_generator.signal(theta2) if theta2 is not None else None

    data = {}
    pure = {}
    for ifo, s1 in signal1["waveform"].items():
        h = s1
        if signal2 is not None:
            h = h + _linear_time_shift(domain, signal2["waveform"][ifo], float(shift_seconds))
        noise = (
            (np.random.randn(len(s1)) + 1j * np.random.randn(len(s1)))
            * asd[ifo]
            * domain.noise_std
        )
        data[ifo] = domain.update_data(h + noise, low_value=0.0)
        pure[ifo] = domain.update_data(h, low_value=0.0)

    signal1["waveform"] = data
    return signal1, pure


def _inject_and_infer(sampler, injection_generator, theta, seed, num_samples, batch_size,
                      theta2=None, shift_seconds=None, noise_seed=None):
    if noise_seed is not None:
        np.random.seed(noise_seed)
    strain_data, pure_signal = _build_injection(
        injection_generator, theta, theta2=theta2, shift_seconds=shift_seconds
    )
    sampler.context = strain_data
    event_metadata = dict(sampler.event_metadata or {})
    event_metadata["injection_seed"] = seed
    sampler.event_metadata = event_metadata
    sampler.run_sampler(num_samples=num_samples, batch_size=batch_size)
    return sampler.to_result(), pure_signal


def _embed_truth(out_hdf5, theta, seed, theta2=None, second_seed=None, shift_seconds=None):
    # Record derived quantities (chi_eff, ...) alongside the sampled parameters, so a
    # model that infers them has a truth to be compared against.
    theta = with_derived_parameters(theta)
    if theta2 is not None:
        theta2 = with_derived_parameters(theta2)
    with h5py.File(out_hdf5, "a") as h:
        for grp in ("injection_parameters", "injection_parameters_2"):
            if grp in h:
                del h[grp]
        g = h.create_group("injection_parameters")
        for name, value in theta.items():
            g.create_dataset(name, data=float(value))
        g.attrs["injection_seed"] = seed
        if theta2 is not None:
            g2 = h.create_group("injection_parameters_2")
            for name, value in theta2.items():
                g2.create_dataset(name, data=float(value))
            g2.attrs["injection_seed"] = second_seed
            g2.attrs["shift_seconds"] = shift_seconds


def _embed_signal(out_hdf5, pure_signal):
    with h5py.File(out_hdf5, "a") as h:
        if "context/signal" in h:
            del h["context/signal"]
        g = h.create_group("context/signal")
        for det, s in pure_signal.items():
            g.create_dataset(det, data=np.asarray(s))


def _embed_model_info(out_hdf5, label, main_path, init_path):
    """Tag the result with the model that produced it (used as the plot legend)."""
    with h5py.File(out_hdf5, "a") as h:
        h.attrs["model_label"] = label
        h.attrs["model_main"] = _rel(main_path)
        h.attrs["model_init"] = _rel(init_path)


def _write_truth_json(path, seed, theta, theta2=None, second_seed=None, shift_seconds=None,
                      **extra):
    payload = {"seed": seed, "parameters": with_derived_parameters(theta), **extra}
    if theta2 is not None:
        payload["second_signal"] = {
            "seed": second_seed,
            "shift_seconds": shift_seconds,
            "parameters": with_derived_parameters(theta2),
        }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _save_result(result, pure_signal, out_hdf5, theta, seed, label, main_path, init_path,
                 theta2=None, second_seed=None, shift_seconds=None, write_truth_json=True):
    """Write one posterior plus everything needed to interpret it later.

    The truth JSON shares the HDF5's stem, so a sweep produces one truth file per
    result instead of every shift overwriting a single shared file.
    """
    result.to_file(out_hdf5)
    _embed_signal(out_hdf5, pure_signal)
    _embed_truth(out_hdf5, theta, seed, theta2=theta2, second_seed=second_seed,
                 shift_seconds=shift_seconds)
    _embed_model_info(out_hdf5, label, main_path, init_path)
    if write_truth_json:
        truth_json = os.path.splitext(out_hdf5)[0] + ".truth.json"
        _write_truth_json(truth_json, seed, theta, theta2=theta2, second_seed=second_seed,
                          shift_seconds=shift_seconds, model=label)
    return out_hdf5

def run_one_injection(args, two_signals):
    """MODE single / overlap: one injection, one model."""
    mode = "overlap" if two_signals else "single"
    label, main_path, init_path = build_models(args.model)
    main_pm, init_pm = _load_pair(main_path, init_path)
    injection_generator = build_injection_generator(main_pm)
    asd_path, stage = model_asd_path(main_pm)

    shift = args.shift if two_signals else None
    out_dir = mode_dir(mode, label)
    stem = _stem(args.seed, args.distance, shift_seconds=shift, tag=args.tag,
                 extra=_second_tags(args) if two_signals else ())
    os.makedirs(out_dir, exist_ok=True)

    _announce(
        mode, label,
        {"main checkpoint": _rel(main_path),
         "init checkpoint": _rel(init_path),
         f"ASD ({stage})": _rel(asd_path),
         "infers": _loaded_model_params(main_pm)},
        out_dir,
        [f"{stem}.hdf5", f"{stem}.truth.json"],
    )

    theta = _draw_truth(injection_generator, args.seed, args.distance, args.distance_std)
    _print_truth(theta, f"True injection parameters (signal 1, seed={args.seed}):")

    theta2 = second_seed = None
    if two_signals:
        second_seed = args.seed + args.second_seed_offset
        theta2 = _draw_truth(injection_generator, second_seed, args.distance, args.distance_std)
        theta2, notes = _apply_second_overrides(theta2, theta, args)
        _print_truth(theta2, f"\nTrue injection parameters (signal 2, seed={second_seed}, "
                             f"shifted by {args.shift}s):")
        for note in notes:
            print(f"  -> {note}")

    sampler = build_sampler(main_pm, init_pm, num_iterations=args.num_iterations,
                                allow_incomplete_gnpe=args.allow_incomplete_gnpe)

    t0 = time.time()
    result, pure_signal = _inject_and_infer(
        sampler, injection_generator, theta, args.seed, args.num_samples, args.batch_size,
        theta2=theta2, shift_seconds=shift, noise_seed=args.seed,
    )
    out_hdf5 = _save_result(result, pure_signal, os.path.join(out_dir, f"{stem}.hdf5"),
                            theta, args.seed, label, main_path, init_path,
                            theta2=theta2, second_seed=second_seed, shift_seconds=shift)
    print(f"\n{mode} inference took {time.time() - t0:.1f} s -> {_rel(out_hdf5)}")
    return out_hdf5


def _resolve_truth_indices(num_injections):
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is not None:
        return [int(task)]
    return list(range(num_injections))


def _load_truth_from_file(path):
    """Read /injection_parameters back out of a result written by this script."""
    with h5py.File(path, "r") as f:
        if "injection_parameters" not in f:
            raise SystemExit(f"{path} has no /injection_parameters to reuse.")
        return {name: float(value[()]) for name, value in f["injection_parameters"].items()}


def _write_failures(out_dir, failures):
    """Append the failed indices. Appended, not overwritten: a resumed run only
    re-attempts the indices that produced no file, so its list is a continuation."""
    path = os.path.join(out_dir, "failures.txt")
    with open(path, "a") as f:
        for index, message in failures:
            f.write(f"{index:04d}\t{message}\n")
    return path


def _infer_with_retries(sampler, injection_generator, theta, seed, num_samples,
                        batch_size, theta2, shift_seconds, index, max_retries,
                        noise_seed=None):
    """One injection, retried if the flow blows up numerically.

    glasflow's rational-quadratic spline asserts `discriminant >= 0` while
    inverting; in float32 a base sample can land in a bin where that quantity
    is negative by rounding alone. It is a property of the draw, not of the
    injection, so re-running the same truth with fresh randomness almost always
    succeeds. A population run must not die on one such draw.

    A retry re-draws BOTH the noise realization and the flow's base samples,
    the truth stays fixed. Coverage is unaffected -- every injection is still
    one independent prior draw with one independent noise realization -- but
    the retried ones are no longer reproducible from `seed` alone, so they are
    reported.
    """
    for attempt in range(max_retries + 1):
        try:
            return _inject_and_infer(
                sampler, injection_generator, theta, seed, num_samples, batch_size,
                theta2=theta2, shift_seconds=shift_seconds, noise_seed=noise_seed,
            ), attempt
        except (AssertionError, RuntimeError, ValueError) as exc:
            if attempt == max_retries:
                raise
            print(f"[{index:04d}] sampling failed on attempt {attempt + 1}/"
                  f"{max_retries + 1} ({type(exc).__name__}: {exc}); "
                  f"retrying with fresh randomness.", flush=True)
            torch.manual_seed(seed + 1_000_000 * (attempt + 1))
            torch.cuda.manual_seed_all(seed + 1_000_000 * (attempt + 1))
            torch.cuda.empty_cache()


def run_many_injections(args, two_signals):
    """MODE multi / overlap-multi: a population of injections."""
    mode = "overlap-multi" if two_signals else "multi"
    label, main_path, init_path = build_models(args.model)
    main_pm, init_pm = _load_pair(main_path, init_path)
    injection_generator = build_injection_generator(main_pm)
    asd_path, stage = model_asd_path(main_pm)

    shift = args.shift if two_signals else None
    # overlap-multi is further split by shift (and by whatever else is pinned on
    # signal 2): a sweep writes one directory per configuration instead of every
    # one fighting over injection_0000.hdf5.
    out_dir = mode_dir(args.out_name or mode, label, shift_seconds=shift)
    if two_signals:
        out_dir = "_".join([out_dir, *_second_tags(args)])
    os.makedirs(out_dir, exist_ok=True)

    fixed_truth = _load_truth_from_file(args.fixed_truth) if args.fixed_truth else None
    fixed_noise_seed = None if args.fixed_noise_seed < 0 else args.fixed_noise_seed
    if fixed_truth is not None and fixed_noise_seed is not None:
        raise SystemExit("--fixed-truth and --fixed-noise-seed together would make "
                         "all 500 injections identical.")

    indices = _resolve_truth_indices(args.num_injections)
    _announce(
        mode, label,
        {"main checkpoint": _rel(main_path),
         "init checkpoint": _rel(init_path),
         f"ASD ({stage})": _rel(asd_path),
         "infers": _loaded_model_params(main_pm),
         "injections": f"{len(indices)} (seeds {args.base_seed}+i)",
         "varying": ("noise only (truth pinned to "
                     f"{_rel(args.fixed_truth)})" if fixed_truth is not None else
                     f"truth only (noise seed pinned to {fixed_noise_seed})"
                     if fixed_noise_seed is not None else "truth and noise")},
        out_dir,
        [f"injection_{indices[0]:04d}.hdf5 … injection_{indices[-1]:04d}.hdf5"]
        if indices else ["(none)"],
    )

    sampler = build_sampler(main_pm, init_pm, num_iterations=args.num_iterations,
                                allow_incomplete_gnpe=args.allow_incomplete_gnpe)

    loop_start = time.time()
    done = 0
    failures = []
    retried = []
    consecutive_failures = 0
    for i in indices:
        out_hdf5 = os.path.join(out_dir, f"injection_{i:04d}.hdf5")
        if os.path.exists(out_hdf5):
            print(f"[{i:04d}] exists -> {_rel(out_hdf5)}, skipping.", flush=True)
            continue

        seed = args.base_seed + i
        if fixed_truth is not None:
            # Same signal every time; `seed` now only drives the noise draw and
            # the flow's base samples.
            theta = dict(fixed_truth)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        else:
            theta = _draw_reproducible_truth(injection_generator, seed)

        theta2 = second_seed = None
        if two_signals:
            second_seed = seed + args.second_seed_offset
            theta2 = _draw_reproducible_truth(injection_generator, second_seed)
            theta2, _ = _apply_second_overrides(theta2, theta, args)

        t0 = time.time()
        try:
            (result, pure_signal), attempts = _infer_with_retries(
                sampler, injection_generator, theta, seed, args.num_samples,
                args.batch_size, theta2, shift, i, args.max_retries,
                noise_seed=fixed_noise_seed,
            )
        except Exception as exc:
            # One unusable draw must not cost the other 499. Record it and move
            # on; the missing index is visible in the summary and in failures.txt.
            failures.append((i, f"{type(exc).__name__}: {exc}"))
            consecutive_failures += 1
            print(f"[{i:04d}] GIVING UP after {args.max_retries + 1} attempts "
                  f"({type(exc).__name__}: {exc}).", flush=True)
            if consecutive_failures >= args.max_consecutive_failures:
                _write_failures(out_dir, failures)
                raise SystemExit(
                    f"ABORTING: {consecutive_failures} injections failed back to "
                    f"back. That is an environment problem, not a bad draw -- the "
                    f"last error was {type(exc).__name__}: {exc}"
                )
            continue
        consecutive_failures = 0
        if attempts:
            retried.append(i)

        # No per-injection truth JSON: the truth is embedded in the HDF5 and a
        # population run would otherwise double its file count for nothing.
        _save_result(result, pure_signal, out_hdf5, theta, seed, label, main_path, init_path,
                     theta2=theta2, second_seed=second_seed, shift_seconds=shift,
                     write_truth_json=False)

        done += 1
        dt = time.time() - t0
        avg = (time.time() - loop_start) / done
        print(f"[{i:04d}] injection done (seed={seed}) in {dt:.1f}s | "
              f"avg {avg:.1f}s -> {_rel(out_hdf5)}", flush=True)

    print(f"\nSaved {done} injection(s) with truth to {_rel(out_dir)}")
    if retried:
        print(f"{len(retried)} needed a retry (not reproducible from the seed): "
              f"{retried}")
    if failures:
        path = _write_failures(out_dir, failures)
        print(f"{len(failures)} injection(s) FAILED and have no file: "
              f"{[index for index, _ in failures]} -> {_rel(path)}")
    if done == 0 and failures:
        raise SystemExit("No injection succeeded. Exiting non-zero so the job is "
                         "not recorded as COMPLETED with nothing to show for it.")
    return out_dir


def run_real_event(args):
    if CONSTRUCT_DIR not in sys.path:
        sys.path.insert(0, CONSTRUCT_DIR)
    from construct_real_event import RealEventData

    event_file = args.event_file
    if not os.path.isabs(event_file) and not os.path.exists(event_file):
        event_file = os.path.join(REAL_EVENT_INPUT_DIR, os.path.basename(event_file))
    if not os.path.exists(event_file):
        raise FileNotFoundError(
            f"{args.event_file} not found. Create it first, e.g.\n"
            f"  python construct_real_event.py --event GW150914"
        )

    # An event context and a posterior both carry a "context" group, so passing a
    # result file here would silently run inference on its stored strain instead
    # of failing. The dataset_type attribute distinguishes them.
    with h5py.File(event_file, "r") as h:
        dataset_type = h.attrs.get("dataset_type", "")
    if dataset_type != "real_event_data":
        raise ValueError(
            f"{_rel(event_file)} has dataset_type={dataset_type!r}, expected "
            f"'real_event_data'. This mode takes an event context written by "
            f"construct_real_event.py (inference_result/events/), not a posterior."
        )

    label, main_path, init_path = build_models(args.model)
    main_pm, init_pm = _load_pair(main_path, init_path)

    dataset = RealEventData(file_name=event_file)
    event_metadata = dict(dataset.event_metadata or {})
    event_name = event_metadata.get(
        "event_name", os.path.splitext(os.path.basename(event_file))[0]
    )

    out_dir = mode_dir("real", label)
    os.makedirs(out_dir, exist_ok=True)
    # "_posterior" so the name still says what it is once it is away from this
    # directory -- the input context is events/<event>.hdf5, and a bare
    # <event>.hdf5 on both sides is too easy to confuse.
    out_name = f"{event_name}_posterior.hdf5"
    out_hdf5 = os.path.join(out_dir, out_name)

    _announce(
        "real", label,
        {"main checkpoint": _rel(main_path),
         "init checkpoint": _rel(init_path),
         "infers": _loaded_model_params(main_pm),
         "event context": _rel(event_file),
         "detectors": ", ".join(sorted(dataset.context["waveform"])),
         "trigger GPS": str(event_metadata.get("time_event"))},
        out_dir, [out_name],
    )

    sampler = build_sampler(main_pm, init_pm, num_iterations=args.num_iterations,
                                allow_incomplete_gnpe=args.allow_incomplete_gnpe)
    sampler.context = dataset.context
    sampler.event_metadata = event_metadata

    t0 = time.time()
    sampler.run_sampler(num_samples=args.num_samples, batch_size=args.batch_size)
    result = sampler.to_result()

    result.to_file(out_hdf5)
    _embed_model_info(out_hdf5, label, main_path, init_path)
    print(f"Real-event inference took {time.time() - t0:.1f} s -> {_rel(out_hdf5)}")
    return out_hdf5


def _prior_spec(pm):
    """The prior a model was trained with (extrinsic overrides intrinsic)."""
    return {
        **pm.metadata["dataset_settings"]["intrinsic_prior"],
        **get_extrinsic_prior_dict(pm.metadata["train_settings"]["data"]["extrinsic_prior"]),
    }


def _check_model_compatible(label, main_pm, ref_prior, ref_domain, theta):
    ok = True

    domain = build_domain_from_model_metadata(main_pm.metadata)
    for attr in ("f_min", "f_max", "delta_f"):
        ref_v, v = getattr(ref_domain, attr, None), getattr(domain, attr, None)
        if ref_v != v:
            print(f"  !! [{label}] domain {attr}: {v} != reference {ref_v}")
            ok = False
    if len(domain) != len(ref_domain):
        print(f"  !! [{label}] domain length {len(domain)} != {len(ref_domain)}")
        ok = False

    prior_spec = _prior_spec(main_pm)
    differing = [k for k in set(prior_spec) | set(ref_prior)
                 if str(prior_spec.get(k)) != str(ref_prior.get(k))]
    if differing:
        print(f"  !  [{label}] prior differs from reference on: {sorted(differing)}")
        for k in sorted(differing):
            print(f"       {k}: {prior_spec.get(k)}")

    prior = build_prior_with_defaults(prior_spec)
    outside = []
    for name, value in theta.items():
        if name in prior and not np.isfinite(prior[name].ln_prob(value)):
            outside.append(f"{name}={value:.4g}")
    if outside:
        print(f"  !! [{label}] truth outside this model's prior support: "
              f"{', '.join(outside)} -- its posterior cannot cover the truth.")
        ok = False

    inference_parameters = main_pm.metadata["train_settings"]["data"]["inference_parameters"]
    missing = sorted(set(inference_parameters) - set(theta))
    if missing:
        print(f"  !  [{label}] infers parameters absent from the truth: {missing}")

    return ok


def run_model_comparison(args):
    specs = _parse_model_specs(args.models)
    ref_label = specs[0][0]

    # Comparison results are keyed by the injection, not by one model, so they
    # live under compare/<injection>/ with one file per model.
    stem = _stem(args.seed, args.distance, tag=args.tag)
    out_dir = os.path.join(POSTERIOR_DIR, "compare", stem)
    os.makedirs(out_dir, exist_ok=True)

    main_pm, init_pm = _load_pair(specs[0][1], specs[0][2])
    injection_generator = build_injection_generator(main_pm)
    asd_path, stage = model_asd_path(main_pm)

    _announce(
        "compare", f"{len(specs)} models, reference={ref_label}",
        {"models": ", ".join(s[0] for s in specs),
         "reference main": _rel(specs[0][1]),
         f"ASD ({stage})": _rel(asd_path),
         "injection": f"seed={args.seed}, distance={args.distance or 'prior'}"},
        out_dir,
        [f"{s[0]}.hdf5" for s in specs] + ["truth.json"],
    )

    theta = _draw_truth(injection_generator, args.seed, args.distance, args.distance_std)
    _print_truth(theta, f"\nTrue injection parameters (seed={args.seed}):")

    # Build the strain ONCE: same signal, same ASD, same noise for every model.
    np.random.seed(args.seed)
    strain_data, pure_signal = _build_injection(injection_generator, theta)

    ref_prior = _prior_spec(main_pm)
    ref_domain = build_domain_from_model_metadata(main_pm.metadata)

    outputs = []
    for i, (label, main_path, init_path) in enumerate(specs):
        if i > 0:
            main_pm, init_pm = _load_pair(main_path, init_path)
        print(f"\n[{label}] {_rel(main_path)}")
        _check_model_compatible(label, main_pm, ref_prior, ref_domain, theta)

        sampler = build_sampler(main_pm, init_pm, num_iterations=args.num_iterations,
                                allow_incomplete_gnpe=args.allow_incomplete_gnpe)
        # deepcopy: the context setter pops 'parameters' out of the dict, so a
        # shared dict would silently lose the truth for every model after the first.
        sampler.context = copy.deepcopy(strain_data)
        event_metadata = dict(sampler.event_metadata or {})
        event_metadata["injection_seed"] = args.seed
        sampler.event_metadata = event_metadata

        # Reseed per model so each posterior is reproducible on its own, rather
        # than depending on how much torch RNG the preceding models consumed.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        t0 = time.time()
        sampler.run_sampler(num_samples=args.num_samples, batch_size=args.batch_size)
        result = sampler.to_result()

        out_hdf5 = _save_result(result, pure_signal, os.path.join(out_dir, f"{label}.hdf5"),
                                theta, args.seed, label, main_path, init_path,
                                write_truth_json=False)
        outputs.append(out_hdf5)
        print(f"[{label}] inference took {time.time() - t0:.1f} s -> {_rel(out_hdf5)}")

        del sampler, main_pm, init_pm
        torch.cuda.empty_cache()

    truth_json = os.path.join(out_dir, "truth.json")
    _write_truth_json(truth_json, args.seed, theta,
                      reference_model=ref_label, models=[s[0] for s in specs])
    print(f"\nSaved truth -> {_rel(truth_json)}")
    # Explicit --dingo paths, not a target name: compare mode writes to
    # posteriors/compare/<injection>/, outside the <model>/<kind>/ layout that
    # plot_posteriors.py discovers by name.
    print("Plot with:\n  python inference_result/analysis/plot_posteriors.py --dingo "
          + " ".join(_rel(o) for o in outputs)
          + " -p chirp_mass mass_ratio luminosity_distance")
    return outputs


def list_models(args=None):
    print(f"DEFAULT_MODEL = {DEFAULT_MODEL!r}   (top of inference.py)")
    env_model = os.environ.get("MODEL")
    if env_model:
        print(f"$MODEL        = {env_model!r}   (overrides DEFAULT_MODEL)")
    print(f"\n{'label':26s} {'':2s} checkpoints (relative to training/) and what they infer")
    print("-" * 96)
    for label, (main_rel, init_rel) in MODEL_REGISTRY.items():
        main_path = os.path.join(TRAINING_DIR, main_rel)
        main_ok = os.path.exists(main_path)
        init_ok = os.path.exists(os.path.join(TRAINING_DIR, init_rel))
        mark = "ok" if (main_ok and init_ok) else "!!"
        print(f"{label:26s} {mark:2s} main {main_rel}{'' if main_ok else '   <- MISSING'}")
        print(f"{'':26s} {'':2s} init {init_rel}{'' if init_ok else '   <- MISSING'}")
        if not main_ok:
            continue
        info = checkpoint_summary(main_path)
        if "error" in info:
            print(f"{'':26s} {'':2s}      (could not read metadata: {info['error']})")
            continue
        params = info["inference_parameters"]
        print(f"{'':26s} {'':2s}      epoch {info['epoch']}, {len(params)} parameters: "
              f"{', '.join(params)}")
        print(f"{'':26s} {'':2s}      {', '.join(info['detectors'])}, "
              f"f_min={info['f_min']:g} f_max={info['f_max']:g}, "
              f"ASD ({info['last_stage']}) {info['asd_dataset_path']}")
        if not info["gnpe_full_loop"]:
            missing = sorted({"ra", "dec", "geocent_time"} - set(params))
            print(f"{'':26s} {'':2s}      !! missing {', '.join(missing)} -> "
                  "GNPE cannot iterate; --allow-incomplete-gnpe permits an "
                  "APPROXIMATE single pass")
    print("-" * 96)
    print("Use:  python inference.py single --model <label>")
    print("      python inference.py single --model 'label=main.pt|init.pt'")
    return None


MODE_DISPATCH = {
    "models": list_models,
    "single": lambda a: run_one_injection(a, two_signals=False),
    "overlap": lambda a: run_one_injection(a, two_signals=True),
    "multi": lambda a: run_many_injections(a, two_signals=False),
    "overlap-multi": lambda a: run_many_injections(a, two_signals=True),
    "real": run_real_event,
    "compare": run_model_comparison,
}


def main(argv=None):
    args = parse_args(argv)
    return MODE_DISPATCH[args.mode](args)


if __name__ == "__main__":
    main()
