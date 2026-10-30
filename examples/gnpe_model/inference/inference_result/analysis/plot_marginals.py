#!/usr/bin/env python
import argparse
import os
import sys

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dingo.gw.result import Result

# The KDE lives in the bilby baseline's plot_corner.py; importing it keeps the
# Dingo-side and bilby-side figures using literally the same density estimate.
sys.path.insert(0, "/srv/beegfs/dpnc/groups/gw/Peng/projects/dingo/bilby_baseline")
from plot_corner import LATEX, density_curve  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(os.path.dirname(HERE), "figures")

DEFAULT_PARAMETERS = ("chirp_mass", "mass_ratio", "luminosity_distance",
                      "geocent_time", "ra", "dec", "psi")

# Dingo stores geocent_time relative to t_ref, so it needs its own label here
# rather than plot_corner's (which is written for absolute-GPS bilby samples).
LABELS = {**LATEX, "geocent_time": r"$t_c - t_{\rm ref}\,[{\rm s}]$"}


def load(path):
    """Samples, truth and signal 2's description for one Dingo result file."""
    result = Result(file_name=path)
    with h5py.File(path, "r") as h:
        truth = ({k: float(h["injection_parameters"][k][()])
                  for k in h["injection_parameters"]}
                 if "injection_parameters" in h else {})
        second = None
        if "injection_parameters_2" in h:
            g2 = h["injection_parameters_2"]
            second = {k: float(g2[k][()]) for k in g2}
            second["shift_seconds"] = float(g2.attrs["shift_seconds"])
            # Derived quantities run_series.py records on the file itself; they
            # are not parameters of signal 2, but they are what a sweep varies.
            for key in ("sky_separation_deg", "source_separation_mpc",
                        "snr2", "snr_ratio"):
                if key in h.attrs:
                    second[key] = float(h.attrs[key])
            if "series_axis" in h.attrs:
                axis = h.attrs["series_axis"]
                second["axis"] = (axis.decode() if isinstance(axis, bytes)
                                  else str(axis))
    return result.samples, truth, second


# What a distraction signal can differ by, and how to say it in a legend. Order
# matters: the source separation is checked before the distance and sky position
# it is built from, so a separation sweep is labelled by the axis it swept and
# not by the two quantities that had to move to produce it.
VARIABLES = (
    ("snr_ratio", lambda s: rf"$\mathrm{{SNR}}_1/\mathrm{{SNR}}_2$ = "
                            rf"{s['snr_ratio']:.3g}"),
    ("shift_seconds", lambda s: rf"$|\Delta t|$ = {s['shift_seconds']:g} s"),
    # The separation sweep is labelled by the angle, which is the knob that was
    # turned; the 3-D distance D it implies is a consequence of that angle and of
    # the distance solve, and naming both invites the reader to treat D as the
    # axis. D stays on the file for anyone who wants to re-bin by it.
    ("sky_separation_deg",
     lambda s: rf"$\gamma$ = {s['sky_separation_deg']:.0f}$^\circ$"),
    ("chirp_mass", lambda s: rf"$\mathcal{{M}}_2$ = {s['chirp_mass']:g} $M_\odot$"),
    ("luminosity_distance",
     lambda s: rf"$d_{{L,2}}$ = {s['luminosity_distance']:g} Mpc"),
    ("mass_ratio", lambda s: rf"$q_2$ = {s['mass_ratio']:.3g}"),
    # Sky position of the intruder, and nothing else: everything about signal 2
    # other than the coordinate being swept is held fixed, so the label names
    # only that coordinate.
    ("ra", lambda s: rf"$\alpha_2$ = {s['ra']:.2f} rad"),
    ("dec", lambda s: rf"$\delta_2$ = {s['dec']:+.2f} rad"),
)

# Every sweep moves more than one number -- pinning the SNR ratio moves the
# distance and the separation with it -- so the label has to follow the axis that
# was CONTROLLED, not every quantity that ended up differing. run_series.py
# records that axis on the file; only files written before it did need the
# fallback below.
AXIS_VARIABLE = {
    "snr_ratio": "snr_ratio",
    "shift": "shift_seconds",
    "chirp_mass": "chirp_mass",
    "luminosity_distance": "luminosity_distance",
    "separation": "sky_separation_deg",
    "ra": "ra",
    "dec": "dec",
}


def _declared_axis(present):
    """The axis these files agree they belong to, if they say and they agree."""
    axes = {s["axis"] for s in present if "axis" in s}
    return axes.pop() if len(axes) == 1 else None


def _separation_is_the_axis(present):
    """Fallback for files with no recorded axis: D moves but SNR_2 does not."""
    snr2 = {round(s["snr2"], 3) for s in present if "snr2" in s}
    return len(snr2) == 1


def auto_labels(paths, seconds):
    """Legend text naming whatever actually differs between these files.

    A sweep over one variable holds the others fixed, so labelling by the fixed
    ones would print the same string on every curve -- which is exactly what
    happens if the chirp-mass files are labelled by their (shared) separation.
    """
    present = [s for s in seconds if s is not None]
    declared = AXIS_VARIABLE.get(_declared_axis(present) or "")
    if declared:
        keep = {declared}
    else:
        # On a separation sweep the distance is what had to move to produce D,
        # not a second axis; everywhere else D is a relabelling of the distance.
        skip = ({"luminosity_distance"} if _separation_is_the_axis(present)
                else {"source_separation_mpc", "snr_ratio"})
        keep = {key for key, _ in VARIABLES} - skip
    varying = [(key, fmt) for key, fmt in VARIABLES
               if key in keep
               and len({round(s[key], 9) for s in present if key in s}) > 1]
    if not varying:  # a single overlap file, or one that differs in nothing
        varying = [v for v in VARIABLES[:1]]

    labels = []
    for path, second in zip(paths, seconds):
        if second is None:
            labels.append("no signal 2")
            continue
        parts = [fmt(second) for key, fmt in varying if key in second]
        labels.append(", ".join(parts)
                      or os.path.splitext(os.path.basename(path))[0])
    return labels


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", nargs="+", help="Dingo result .hdf5 files.")
    p.add_argument("-p", "--parameters", nargs="+", default=None,
                   help=f"Default: {', '.join(DEFAULT_PARAMETERS)} (those present).")
    p.add_argument("--labels", nargs="+", default=None,
                   help="Legend labels, one per file. Default: the time separation.")
    p.add_argument("-o", "--out", default=None,
                   help="Output file. Default: figures/series_marginals.png")
    p.add_argument("--bw", type=float, default=1.0,
                   help="KDE bandwidth scale (>1 = smoother).")
    p.add_argument("--ncols", type=int, default=4)
    p.add_argument("--central", type=float, default=99.8, metavar="PERCENT",
                   help="Percent of samples the x range covers; the rest are "
                        "tail outliers that would squash the panel. 100 = full range.")
    p.add_argument("--no-truth", action="store_true",
                   help="Do not draw the injected value of signal 1.")
    p.add_argument("--peak-normalise", "--peak-normalize", action="store_true",
                   dest="peak_normalise",
                   help="Scale every curve to its own maximum. A posterior that "
                        "locked onto a loud intruder is tens of times narrower "
                        "than the rest, and on a shared density axis it flattens "
                        "them; this compares shapes at the cost of the areas.")
    p.add_argument("--second-truth", action="store_true",
                   help="Also mark signal 2's injected value (dotted, in the "
                        "curve's colour) -- tells a posterior that has locked "
                        "onto the intruder from one that is merely wider.")
    args = p.parse_args(argv)

    frames, truths, seconds = [], [], []
    for path in args.results:
        samples, truth, second = load(path)
        frames.append(samples)
        truths.append(truth)
        seconds.append(second)
    labels = auto_labels(args.results, seconds)
    for path, samples, label in zip(args.results, frames, labels):
        print(f"loaded {os.path.basename(path)}: {len(samples)} samples, "
              f"label {label!r}")
    if args.labels:
        if len(args.labels) != len(frames):
            raise SystemExit(f"--labels has {len(args.labels)} entries for "
                             f"{len(frames)} files")
        labels = args.labels

    parameters = args.parameters or [q for q in DEFAULT_PARAMETERS
                                     if q in frames[0].columns]
    missing = [q for q in parameters if any(q not in f.columns for f in frames)]
    if missing:
        raise SystemExit(f"parameter(s) {missing} are not in every file. "
                         f"Available: {sorted(frames[0].columns)}")

    # Signal 1 is meant to be identical in every file; say so if it is not,
    # because then no single truth line is meaningful.
    truth = truths[0]
    for other, path in zip(truths[1:], args.results[1:]):
        differing = [k for k in truth
                     if k in other and not np.isclose(truth[k], other[k])]
        if differing:
            print(f"  ! {os.path.basename(path)} has a different signal 1 "
                  f"({differing}); the truth lines follow the first file.")

    ncols = min(args.ncols, len(parameters))
    nrows = int(np.ceil(len(parameters) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows),
                             squeeze=False)

    for index, param in enumerate(parameters):
        ax = axes[index // ncols][index % ncols]
        values = [f[param].to_numpy() for f in frames]
        # A single stray sample can otherwise set the axis and squash the peak
        # into a spike -- geocent_time did exactly that at full range.
        tail = 0.5 * (100.0 - args.central)
        lo = min(np.percentile(v, tail) for v in values)
        hi = max(np.percentile(v, 100.0 - tail) for v in values)
        if not np.isfinite(hi - lo) or hi <= lo:
            lo, hi = min(v.min() for v in values), max(v.max() for v in values)
        pad = 0.04 * (hi - lo)
        drawn_second = set()
        for series, label, second in zip(values, labels, seconds):
            grid, density = density_curve(series, (lo - pad, hi + pad), bw=args.bw)
            if args.peak_normalise and density.max() > 0:
                density = density / density.max()
            line, = ax.plot(grid, density, lw=1.6, label=label)
            if args.second_truth and second is not None and param in second:
                # A sweep holds signal 2's other parameters fixed, so the same
                # line would otherwise be drawn once per curve.
                key = round(second[param], 9)
                if key not in drawn_second:
                    drawn_second.add(key)
                    ax.axvline(second[param], color=line.get_color(), ls=":",
                               lw=1.4,
                               label="signal 2 truth" if index == 0
                               and len(drawn_second) == 1 else None)
        if not args.no_truth and param in truth:
            ax.axvline(truth[param], color="black", ls="--", lw=1.2,
                       label="truth" if index == 0 else None)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(LABELS.get(param, param))
        ax.set_ylabel(("posterior density"
                       + (" (peak = 1)" if args.peak_normalise else ""))
                      if index % ncols == 0 else "")
        ax.set_yticklabels([])

    for index in range(len(parameters), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    # Outside the axes, to the right: an inset legend covers the tail of the
    # rightmost panel, which is exactly where curves are compared. bbox_inches
    # ="tight" plus bbox_extra_artists below grows the canvas to include it.
    handles, legend_labels = axes[0][0].get_legend_handles_labels()
    # Two-line entries run into each other at the default spacing.
    spacing = 1.1 if any("\n" in text for text in legend_labels) else 0.5
    legend = fig.legend(handles, legend_labels, loc="center left",
                        bbox_to_anchor=(1.0, 0.5), fontsize=13, frameon=False,
                        labelspacing=spacing)
    fig.tight_layout()

    out = args.out or os.path.join(FIGURES_DIR, "series_marginals.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight", bbox_extra_artists=(legend,))
    plt.close(fig)
    print(f"\nSaved -> {out}")
    return out


if __name__ == "__main__":
    main()
