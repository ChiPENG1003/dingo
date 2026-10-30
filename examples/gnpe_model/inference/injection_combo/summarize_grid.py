"""Turn a sweep's per-run records into the 10 x 10 coverage grids the plots read.

A cell's value is n_covered / n_total: the fraction of that cell's injections
whose TRUE signal-1 parameter fell inside the central credible interval of the
recovered posterior. The denominator is the number of runs that actually
finished, never a hard-coded 100 -- a cell that lost tasks must not be able to
show up as a confident colour.

Because every run stores the truth's posterior quantile rather than a verdict,
the credible level is a plotting choice made here, not at injection time.

The `--bin-by` option re-bins a finished sweep by any quantity that was
recorded, which is how the distance heat maps come out of the SNR-ratio sweep
for free:

  python summarize_grid.py snr_ratio                              # by SNR ratio
  python summarize_grid.py snr_ratio --bin-by luminosity_distance # by d_2
  python summarize_grid.py snr_ratio --bin-by delta_distance      # by d_2 - d_1
"""

import argparse
import glob
import os

import h5py
import numpy as np

import combo_common as cc

# Quantities a sweep can be binned by, as functions of one run's record. Anything
# here can serve as an x axis regardless of which axis the sweep was RUN on.
DERIVED = {
    "snr_ratio": lambda r, c: c["target_snr1"] / r["snr2"],
    "luminosity_distance": lambda r, c: r["theta2_luminosity_distance"],
    "delta_distance": lambda r, c: (r["theta2_luminosity_distance"]
                                    - c["theta1"]["luminosity_distance"]),
    "chirp_mass": lambda r, c: r["theta2_chirp_mass"],
    "mass_ratio": lambda r, c: r["theta2_mass_ratio"],
    "snr2": lambda r, c: r["snr2"],
    # Sky separation acts on the data: it changes both antenna patterns and the
    # H1-L1 arrival delay, which is what lets the network tell two signals apart.
    "sky_separation": lambda r, c: cc.sky_separation(
        c["theta1"]["ra"], c["theta1"]["dec"], r["theta2_ra"], r["theta2_dec"]),
    # 3-D comoving separation. Recorded because it was asked for, but it does
    # not enter the likelihood -- see the note in combo_common before reading
    # anything into a flat heat map on this axis.
    # D = d_L * gamma, the transverse arc. Only meaningful when both sources
    # share a distance, which the separation sweep enforces; measured here from
    # signal 2's own d_L so a mismatch would show up rather than be assumed.
    "separation": lambda r, c: cc.transverse_separation(
        r["theta2_luminosity_distance"],
        cc.sky_separation(c["theta1"]["ra"], c["theta1"]["dec"],
                          r["theta2_ra"], r["theta2_dec"])),
    "source_separation": lambda r, c: cc.source_separation(
        c["theta1"]["luminosity_distance"], c["theta1"]["ra"], c["theta1"]["dec"],
        r["theta2_luminosity_distance"], r["theta2_ra"], r["theta2_dec"]),
}

LABELS = {
    "snr_ratio": r"$\mathrm{SNR}_1 / \mathrm{SNR}_2$",
    "luminosity_distance": r"$d_{L,2}$ [Mpc]",
    "delta_distance": r"$d_{L,2} - d_{L,1}$ [Mpc]",
    "chirp_mass": r"$\mathcal{M}_2$ [$M_\odot$]",
    "mass_ratio": r"$q_2$",
    "snr2": r"$\mathrm{SNR}_2$",
    "sky_separation": r"sky separation $\gamma$ [deg]",
    "separation": r"$D = d_L\,\gamma$ [Mpc]",
    "source_separation": r"comoving source separation [Mpc]",
}


def load_runs(sweep_name):
    """Every run of a sweep as a list of flat dicts, plus the sweep config."""
    config = cc.read_json(os.path.join(cc.sweep_dir(sweep_name), "config.json"))
    rows = []
    pattern = os.path.join(cc.sweep_dir(sweep_name), "cells", "x*_y*.hdf5")
    for path in sorted(glob.glob(pattern)):
        with h5py.File(path, "r") as f:
            cell = {k: f.attrs[k] for k in ("x_index", "y_index", "x_value",
                                            "shift_seconds")}
            for name in f:
                if not name.startswith("run_"):
                    continue
                row = dict(cell)
                row["realization"] = int(name.split("_")[1])
                row.update({k: v for k, v in f[name].attrs.items()})
                rows.append(row)
    if not rows:
        raise SystemExit(f"No runs found under {cc.rel(cc.sweep_dir(sweep_name))}/cells/. "
                         f"Has the sweep been launched?")
    return config, rows


def reference_coverage(level=0.9, path=None):
    """Coverage of the single-signal control, per parameter.

    This -- not the credible level -- is what a cell has to be compared with.
    The study fixes ONE target signal, and a coverage test is only calibrated to
    its nominal level when the truth is redrawn from the prior each time; with a
    fixed truth the quantiles are not uniform even for a perfect sampler. The
    control shares that truth and those noise seeds and simply omits the
    distraction signal, so the difference between a cell and the control is
    attributable to the overlap. Returns {} if the control has not been run.
    """
    path = path or os.path.join(cc.REFERENCE_DIR, "reference.hdf5")
    if not os.path.exists(path):
        return {}
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    quantiles = {}
    with h5py.File(path, "r") as f:
        for name in f:
            if not name.startswith("run_"):
                continue
            for key, value in f[name].attrs.items():
                if key.startswith("quantile_"):
                    quantiles.setdefault(key[len("quantile_"):], []).append(float(value))
    return {p: float(np.mean((np.array(v) > lo) & (np.array(v) < hi)))
            for p, v in quantiles.items()}


def derived_edges(values, num_x, mode="quantile"):
    """x-axis boundaries for a quantity that was measured rather than controlled.

    'quantile' gives equal-count bins: column widths then show where the
    population actually lives, which is the honest way to ask "how often does
    this happen". Its cost is that a skewed quantity spends most of its columns
    in one regime -- the natural SNR ratio has a median of 3.7 and only 10% of
    draws below 1, so 9 of 10 equal-count columns land in the region the
    controlled sweep already showed to be harmless.

    'log' gives fixed log-spaced edges over the 1st-99th percentile: every
    column is then physically distinct, at the price of very unequal counts,
    which the plot marks as short cells.
    """
    finite = values[np.isfinite(values)]
    if mode == "log":
        lo, hi = np.percentile(finite, [1.0, 99.0])
        lo = max(lo, 1e-12)
        edges = np.geomspace(lo, hi, num_x + 1)
        edges[0], edges[-1] = finite.min(), np.nextafter(finite.max(), np.inf)
        return edges
    edges = np.quantile(finite, np.linspace(0, 1, num_x + 1))
    edges[-1] = np.nextafter(edges[-1], np.inf)
    return edges


def coverage_grid(rows, config, parameter, level=0.9, bin_by=None, num_x=None,
                  bin_mode="quantile"):
    """(coverage, n_covered, n_total, x_edges) for one parameter.

    Covered means the truth's quantile lies inside the central `level` interval.
    """
    key = f"quantile_{parameter}"
    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0

    y_values = np.asarray(config["time_shifts"], dtype=float)
    num_y = len(y_values)

    assigned = None
    if bin_by is None:
        # The sweep's own axis: cells are already the bins.
        num_x = num_x or len(config["axis_centers"])
        x_edges = np.asarray(config["axis_edges"], dtype=float)
    else:
        if bin_by not in DERIVED:
            raise SystemExit(f"Cannot bin by {bin_by!r}. Known: {sorted(DERIVED)}")
        values = np.array([DERIVED[bin_by](r, config) for r in rows], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise SystemExit(f"No finite values of {bin_by!r} in this sweep.")
        num_x = num_x or len(config["axis_centers"])
        x_edges = derived_edges(values, num_x, mode=bin_mode)
        assigned = np.clip(np.searchsorted(x_edges, values, side="right") - 1,
                           0, num_x - 1)

    n_total = np.zeros((num_y, num_x), dtype=int)
    n_covered = np.zeros((num_y, num_x), dtype=int)

    for i, row in enumerate(rows):
        if key not in row:
            continue
        x = int(row["x_index"]) if assigned is None else int(assigned[i])
        y = int(row["y_index"])
        if not (0 <= x < num_x and 0 <= y < num_y):
            continue
        n_total[y, x] += 1
        if lo < float(row[key]) < hi:
            n_covered[y, x] += 1

    with np.errstate(invalid="ignore", divide="ignore"):
        coverage = np.where(n_total > 0, n_covered / np.maximum(n_total, 1), np.nan)
    return coverage, n_covered, n_total, x_edges, y_values


def summarize(sweep_name, parameters, level=0.9, bin_by=None, out_name=None,
              bin_mode="quantile", mean_over=None):
    config, rows = load_runs(sweep_name)
    axis_name = bin_by or config["axis"]
    grids = {}
    for parameter in parameters:
        coverage, covered, total, x_edges, y_values = coverage_grid(
            rows, config, parameter, level=level, bin_by=bin_by, bin_mode=bin_mode
        )
        grids[parameter] = (coverage, covered, total)
        threshold = cc.short_cell_threshold(total, config["num_realizations"],
                                            bin_by is not None)
        empty = int((total == 0).sum())
        short = int(((total > 0) & (total < threshold)).sum())
        print(f"  {parameter:22s} mean coverage {np.nanmean(coverage):.3f}   "
              f"min {np.nanmin(coverage):.3f}   "
              f"{empty} empty cell(s), {short} short cell(s)")

    members = []
    if mean_over:
        # Mean coverage across parameters. Every parameter shares the same
        # n_total in a cell, so averaging the coverages is the same as pooling
        # n_covered -- no weighting decision is hidden here.
        members = [p for p in mean_over if p in grids]
        if members:
            stack = np.stack([grids[p][0] for p in members])
            covered = np.stack([grids[p][1] for p in members]).sum(axis=0)
            total = grids[members[0]][2] * len(members)
            grids["mean"] = (np.nanmean(stack, axis=0), covered, total)
            print(f"  {'mean':22s} mean coverage {np.nanmean(grids['mean'][0]):.3f}   "
                  f"min {np.nanmin(grids['mean'][0]):.3f}   "
                  f"over {len(members)}: {', '.join(members)}")

    baseline = reference_coverage(level=level)
    if "mean" in grids and baseline:
        # The SAME members the grid averaged over. Recomputing the list against
        # the baseline's own keys would silently average the control over all
        # seven parameters while the grid averaged over four, and the two
        # numbers would be printed side by side as if comparable.
        missing = [p for p in members if p not in baseline]
        if missing:
            print(f"  !! no control coverage for {missing}; the mean baseline "
                  f"would not match the mean grid, so it is omitted.")
        else:
            baseline["mean"] = float(np.mean([baseline[p] for p in members]))
    if baseline:
        print("  no-intruder baseline: " +
              ", ".join(f"{p} {baseline[p]:.2f}" for p in grids if p in baseline))
    else:
        print("  !! no single-signal control found -- run "
              "'sbatch submit_grid.sh reference'. Without it the cell values "
              "have nothing to be compared against.")

    suffix = "" if (bin_by is None or bin_mode == "quantile") else f"_{bin_mode}"
    out_name = out_name or f"coverage_{axis_name}{suffix}.npz"
    out_path = os.path.join(cc.sweep_dir(sweep_name), out_name)
    np.savez(
        out_path,
        baseline=np.array([baseline], dtype=object),
        x_edges=x_edges,
        y_values=y_values,
        x_axis=axis_name,
        x_label=LABELS.get(axis_name, axis_name),
        sweep=sweep_name,
        level=level,
        parameters=np.array(list(grids), dtype=object),
        coverage=np.stack([grids[p][0] for p in grids]),
        n_covered=np.stack([grids[p][1] for p in grids]),
        n_total=np.stack([grids[p][2] for p in grids]),
        mean_over=np.array(list(mean_over or []), dtype=object),
        num_realizations=config["num_realizations"],
        snr1=config["target_snr1"],
        theta1=np.array([config["theta1"]], dtype=object),
        binned_by_derived=bin_by is not None,
    )
    print(f"-> {cc.rel(out_path)}")
    return out_path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep")
    p.add_argument("--parameters", nargs="+", default=None,
                   help="Defaults to every parameter the model infers.")
    p.add_argument("--level", type=float, default=0.9,
                   help="Credible level the coverage is measured at.")
    p.add_argument("--bin-by", default=None, choices=sorted(DERIVED),
                   help="Re-bin the x axis by a recorded quantity instead of the "
                        "axis the sweep was run on. Uses equal-count bins.")
    p.add_argument("--bin-mode", default="quantile", choices=("quantile", "log"),
                   help="quantile = equal-count columns (shows where the "
                        "population is); log = fixed log-spaced columns (every "
                        "column physically distinct, unequal counts).")
    p.add_argument("--mean", action="store_true",
                   help="Also emit a 'mean' grid: coverage averaged over the "
                        "parameters. Note psi and mass_ratio have essentially "
                        "prior-wide posteriors and sit at 1.00 coverage, so "
                        "including them puts a floor under every cell; "
                        "--mean-over lets you drop them.")
    p.add_argument("--mean-over", nargs="+", default=None,
                   help="Parameters the mean runs over. Defaults to every "
                        "parameter the model infers.")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    config = cc.read_json(os.path.join(cc.sweep_dir(args.sweep), "config.json"))
    parameters = args.parameters or config["inference_parameters"]
    print(f"sweep {args.sweep}  axis {args.bin_by or config['axis']}  "
          f"level {args.level:g}  SNR_1 {config['target_snr1']:.2f}")
    mean_over = None
    if args.mean or args.mean_over:
        mean_over = args.mean_over or list(config["inference_parameters"])
    return summarize(args.sweep, parameters, level=args.level, bin_by=args.bin_by,
                     out_name=args.out, bin_mode=args.bin_mode, mean_over=mean_over)


if __name__ == "__main__":
    main()
