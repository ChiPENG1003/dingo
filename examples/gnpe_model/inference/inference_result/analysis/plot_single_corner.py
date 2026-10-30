#!/usr/bin/env python
import argparse
import glob
import os
import re

import h5py

from dingo.core.utils.plotting import get_latex_labels, plot_corner_multi
from dingo.gw.result import Result

# This file lives in inference_result/analysis/. Everything it reads and writes
# is a sibling of that directory, so anchor to inference_result rather than to
# the analysis folder itself.
HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.dirname(HERE)
EVENTS_DIR = os.path.join(RESULT_ROOT, "events")        # observed strain (inputs)
LEGACY_DIR = os.path.join(RESULT_ROOT, "legacy")        # results from before the reorg
FIGURES_DIR = os.path.join(RESULT_ROOT, "figures")
TABLES_DIR = os.path.join(RESULT_ROOT, "tables")

# All figures are written here (created on demand).
PICTURE_DIR = FIGURES_DIR


def _picture_path(name):
    """Return <HERE>/picture/<name>, creating the folder if needed."""
    os.makedirs(PICTURE_DIR, exist_ok=True)
    return os.path.join(PICTURE_DIR, name)

# Non-physical / internal sample columns that should never appear in the corner.
_NON_PHYSICAL = {
    "log_prob", "log_likelihood", "log_prior", "weights",
    "delta_log_prob_target",
}


def _default_corner_name(input_path):
    base = os.path.splitext(os.path.basename(input_path))[0]

    m = re.search(r"_?overlap_?(\d+(?:\.\d+)?)?$", base)
    if m:
        stem = base[: m.start()].rstrip("_")
        n = m.group(1)
        suffix = f"_{n}" if n is not None else "_overlap"
        name = f"{stem}_corner{suffix}.png"
    else:
        name = f"{base}_corner.png"

    return _picture_path(name)


def _normalize_parameters(parameters):
    """Return a flat list of parameter names, splitting comma-joined entries.

    Accepts a bare string, a list of names, or a list where some entries contain
    commas (e.g. ["chirp_mass, mass_ratio"] -> ["chirp_mass", "mass_ratio"]).
    """
    if isinstance(parameters, str):
        parameters = [parameters]
    flat = []
    for p in parameters:
        flat.extend(s.strip() for s in p.split(",") if s.strip())
    return flat


def _shift_of(path):
    """Parse the overlap shift from a filename; None if not an overlap file."""
    base = os.path.splitext(os.path.basename(path))[0]
    m = re.search(r"_?overlap_?(\d+(?:\.\d+)?)?$", base)
    if not m:
        return None
    return m.group(1)  # e.g. "7.0", or None for a bare "_overlap"


def _model_label_of(path):
    """The model tag written by inference.py's compare mode, if present."""
    try:
        with h5py.File(path, "r") as f:
            label = f.attrs.get("model_label")
    except OSError:
        return None
    if label is None:
        return None
    return label.decode() if isinstance(label, bytes) else str(label)


def _label_from_path(path):
    """Legend label: model name if the file has one, else overlap shift."""
    model = _model_label_of(path)
    if model is not None:
        return model

    shift = _shift_of(path)
    base = os.path.splitext(os.path.basename(path))[0]
    if "overlap" not in base:
        return "reference"
    return f"$\\Delta t$ = {shift} s" if shift is not None else "overlap"


def _discover_injections(input_path):
    """Auto-collect injection*.hdf5 next to --input (reference + overlaps)."""
    directory = os.path.dirname(os.path.abspath(input_path)) or HERE
    return glob.glob(os.path.join(directory, "injection*.hdf5"))


def _sorted_injections(files):
    """Reference first, then overlaps by increasing shift.

    Model-comparison files have no meaningful ordering, so they are left in the
    order the caller listed them -- that is the order the legend follows.
    """
    if files and all(_model_label_of(f) is not None for f in files):
        return list(files)

    def key(f):
        shift = _shift_of(f)
        if "overlap" not in os.path.basename(f):
            return (0, 0.0)
        return (1, float(shift) if shift is not None else 0.0)
    return sorted(files, key=key)


def _default_compare_name(files):
    if files and all(_model_label_of(f) is not None for f in files):
        return _picture_path("compare_models_corner.png")
    return _picture_path("compare_corner.png")


def _plot_marginal_multi(samples_list, weights_list, labels, param, out, truth=None):
    """Overlay 1D marginals of `param` from several injections (compare, K=1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s, w, l in zip(samples_list, weights_list, labels):
        ax.hist(s[param].to_numpy(), bins=80, density=True, histtype="step",
                weights=w, label=l)
    if truth is not None and truth.get(param) is not None:
        ax.axvline(truth[param], color="k", ls="--", lw=1.2, label="truth")
    ax.set_xlabel(param)
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_compare(files, parameters, out, labels_override=None):
    samples_list, weights_list, labels = [], [], []
    truth = None
    prior = None
    for f in files:
        result = Result(file_name=f)
        if prior is None:
            prior = result.prior
        if truth is None:
            # Every file here is conditioned on the same injected signal 1, so a
            # single truth vector applies to every curve.
            truth = load_embedded_truth(f)
        samples_list.append(result.samples[parameters])
        weights = (
            result.samples["weights"].to_numpy()
            if "weights" in result.samples else None
        )
        weights_list.append(weights)
        labels.append(_label_from_path(f))
        print(f"  + {os.path.basename(f):32s} "
              f"{len(result.samples):>7d} samples  label='{labels[-1]}'")

    if labels_override:
        if len(labels_override) != len(files):
            raise ValueError(
                f"--labels has {len(labels_override)} entries but "
                f"{len(files)} files were given."
            )
        labels = list(labels_override)

    kwargs = {"legend_font_size": 14}  # dingo defaults to 50, which swamps the figure
    if truth is not None:
        kwargs["truths"] = [truth.get(p) for p in parameters]

    if len(parameters) == 1:
        _plot_marginal_multi(samples_list, weights_list, labels, parameters[0],
                             out, truth)
    else:
        plot_corner_multi(
            samples_list, weights=weights_list, labels=labels, filename=out,
            latex_labels_dict=get_latex_labels(prior), **kwargs,
        )
    print(f"Saved comparison corner ({len(files)} injections) -> {out}")


def plot_single_marginal(result, param, out, truth=None):
    """1D marginal for a single parameter (corner can't do K=1 with a truth line)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = result.samples
    x = samples[param].to_numpy()
    weights = samples["weights"].to_numpy() if "weights" in samples else None

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(x, bins=80, density=True, histtype="step", color="C0", label="Dingo")
    if weights is not None:
        ax.hist(x, bins=80, density=True, weights=weights, histtype="step",
                color="C1", label="Dingo-IS")
    if truth is not None and truth.get(param) is not None:
        ax.axvline(truth[param], color="k", ls="--", lw=1.2, label="truth")
    ax.set_xlabel(param)
    ax.set_ylabel("density")
    ax.legend()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_embedded_truth(path):
    with h5py.File(path, "r") as f:
        if "injection_parameters" not in f:
            return None
        grp = f["injection_parameters"]
        return {k: float(grp[k][()]) for k in grp.keys()}


def load_embedded_distraction(path):
    with h5py.File(path, "r") as f:
        if "injection_parameters_2" not in f:
            return None
        grp = f["injection_parameters_2"]
        return {k: float(grp[k][()]) for k in grp.keys()}


def plot_corner_two_signals(result, parameters, out, truth, distraction,
                            truth_color="black", distraction_color="crimson"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.legend import Legend
    import numpy as np
    import corner

    from dingo.core.utils.plotting import plot_corner_multi, get_latex_labels

    samples = result.samples
    theta = samples[parameters]
    weights = (samples["weights"].to_numpy()
               if "weights" in samples.columns else None)

    if weights is not None:
        samples_list, weights_list, labels = (
            [theta, theta], [None, weights], ["Dingo", "Dingo-IS"])
    else:
        samples_list, weights_list, labels = [theta], [None], ["Dingo"]

    truth_vals = [truth.get(p) if truth else None for p in parameters]

    fig = plot_corner_multi(
        samples_list, weights=weights_list, labels=labels, filename=out,
        latex_labels_dict=get_latex_labels(result.prior),
        truths=truth_vals, truth_color=truth_color,
        legend_font_size=14,
    )

    # Overlay the distraction (second overlapping signal) in a distinct style.
    dist_vals = [distraction.get(p, np.nan) for p in parameters]
    dist_vals = [np.nan if v is None else v for v in dist_vals]
    corner.overplot_lines(fig, dist_vals, color=distraction_color,
                          linestyle="--", linewidth=1.4)
    corner.overplot_points(fig, np.array(dist_vals)[None, :], marker="s",
                           markersize=6, color=distraction_color)

    # Second legend: identify the two vertical/marker lines (truth vs distraction).
    handles = [
        Line2D([], [], color=truth_color, linestyle="-", linewidth=2,
               label="truth (signal 1)"),
        Line2D([], [], color=distraction_color, linestyle="--", linewidth=2,
               label="distraction (signal 2)"),
    ]
    leg = Legend(fig, handles, [h.get_label() for h in handles],
                 loc="upper right", bbox_to_anchor=(0.98, 0.80),
                 fontsize=14, frameon=False)
    fig.add_artist(leg)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def default_parameters(result):
    """Physical, sampled parameters present in the posterior (drop GNPE proxies)."""
    cols = list(result.samples.columns)
    return [
        c for c in cols
        if c not in _NON_PHYSICAL and not c.endswith("_time_proxy")
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-i", "--input", default=os.path.join(HERE, "real_events", "GW170809_result.hdf5"),
                        help="Single-injection Dingo result HDF5 (default: injection.hdf5).")
    parser.add_argument("-o", "--output", default=None,
                        help="Output corner file (default: <input>_corner.png).")
    parser.add_argument("-p", "--parameters", nargs="+",
                        # default=["chirp_mass", "mass_ratio", "luminosity_distance"],
                        help="Parameters to plot (default: chirp_mass, mass_ratio, luminosity_distance).")
    parser.add_argument("-c", "--compare", nargs="*", default=None,
                        help="Overlay MODE: plot several injections on one corner. "
                             "Give a list of HDF5 files, or pass the flag with no "
                             "files to auto-collect injection*.hdf5 next to --input.")
    parser.add_argument("-l", "--labels", nargs="+", default=None,
                        help="Legend labels for compare mode, one per file, in the "
                             "order given. Defaults to the model tag stored in each "
                             "file, else the overlap shift.")
    args = parser.parse_args()

    # Compare / overlay mode: several injections on one figure, same parameters.
    if args.compare is not None:
        parameters = _normalize_parameters(args.parameters)
        files = _sorted_injections(args.compare or _discover_injections(args.input))
        if not files:
            parser.error("compare mode found no injection HDF5 files to plot.")
        out = args.output or _default_compare_name(files)
        print(f"Compare mode: overlaying {len(files)} injections on {parameters}")
        run_compare(files, parameters, out, labels_override=args.labels)
        return

    out = args.output or _default_corner_name(args.input)

    result = Result(file_name=args.input)

    # Make plot_corner mark the injected truth: it looks for
    # event_metadata["injection_parameters"], but inference.py stores the truth in
    # a separate top-level group, so wire it in explicitly.
    truth = load_embedded_truth(args.input)
    if truth is not None:
        meta = dict(result.event_metadata or {})
        meta["injection_parameters"] = truth
        result.event_metadata = meta
        print(f"Injected truth marked for {len(truth)} parameters.")
    else:
        print("WARNING: no /injection_parameters group found; truth will not be marked.")

    # If this is an overlap injection, also mark the second (distraction) signal.
    distraction = load_embedded_distraction(args.input)
    if distraction is not None:
        print(f"Distraction (signal 2) found; marking {len(distraction)} parameters.")

    #parameters = _normalize_parameters(args.parameters or default_parameters(result))
    parameters = _normalize_parameters(default_parameters(result))
    print(f"Plotting {len(parameters)} parameters: {parameters}")

    if len(parameters) == 1:
        plot_single_marginal(result, parameters[0], out, truth)
    elif distraction is not None:
        plot_corner_two_signals(result, parameters, out, truth, distraction)
    else:
        result.plot_corner(parameters=parameters, filename=out)
    print(f"Saved corner plot -> {out}")


if __name__ == "__main__":
    main()
