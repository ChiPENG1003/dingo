#!/usr/bin/env python
import os
import sys
import json
import argparse
from types import SimpleNamespace
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import bilby

DEFAULT_T_REF = 1126259462.391

PARAM_ORDER = [
    "chirp_mass", "mass_ratio", "mass_1", "mass_2", "total_mass",
    "chi_eff", "chi_p", "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl",
    "theta_jn", "luminosity_distance", "psi",
]

LATEX = {
    "chirp_mass": r"$\mathcal{M}\,[M_\odot]$", "mass_ratio": r"$q$",
    "mass_1": r"$m_1\,[M_\odot]$", "mass_2": r"$m_2\,[M_\odot]$",
    "total_mass": r"$M\,[M_\odot]$",
#    "a_1": r"$a_1$", "a_2": r"$a_2$", "tilt_1": r"$\theta_1$",
#    "tilt_2": r"$\theta_2$", "phi_12": r"$\phi_{12}$", "phi_jl": r"$\phi_{JL}$",
#    "chi_eff": r"$\chi_{\rm eff}$", "chi_p": r"$\chi_p$",
#    "theta_jn": r"$\theta_{JN}$",
    "luminosity_distance": r"$d_L\,[{\rm Mpc}]$",
    "ra": r"$\alpha$", "dec": r"$\delta$", "psi": r"$\psi$",
#    "phase": r"$\phi_c$", "geocent_time": r"$t_c - t_{\rm ref}\,[{\rm s}]$",
}

NON_PHYSICAL = {"log_likelihood", "log_prior", "weights", "reference_frequency",
                "waveform_approximant", "minimum_frequency", "log_prob",
                "delta_log_prob_target"}

def resolve_result_path(path):
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for ext in ("hdf5", "json"):
            hits = sorted(f for f in os.listdir(path) if f.endswith(f"_result.{ext}"))
            if hits:
                return os.path.join(path, hits[0])
        raise FileNotFoundError(f"no *_result.json/hdf5 in {path}")
    raise FileNotFoundError(path)


def load_result(path):
    """Load either a Bilby result or a Dingo posterior HDF5 file.

    Bilby HDF5 results contain a ``posterior`` group, while Dingo stores its
    posterior samples in a top-level ``samples`` dataset.  The latter cannot be
    decoded by ``bilby.core.result.read_in_result``.
    """
    if path.endswith((".hdf5", ".h5")):
        import h5py
        with h5py.File(path, "r") as f:
            is_dingo = "samples" in f and "posterior" not in f
        if is_dingo:
            from dingo.gw.result import Result

            result = Result(file_name=path)
            posterior = result.samples.copy()
            # Time-proxy columns are Dingo internal features, not source
            # parameters.  Keeping them out also gives useful default panels.
            parameters = [p for p in posterior.columns
                          if p not in NON_PHYSICAL and not p.endswith("_time_proxy")]
            return SimpleNamespace(
                posterior=posterior,
                search_parameter_keys=parameters,
                priors=getattr(result, "prior", None),
                t_ref=getattr(result, "t_ref", None),
                # Dingo samples are already offsets from this reference time.
                time_is_relative=True,
            )
    result = bilby.core.result.read_in_result(filename=path)
    # Keep one interface for the plotting code below.
    result.t_ref = None
    result.time_is_relative = False
    return result


def load_truth(path):
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        # plot_single_corner.py's truth json nests the vector under "parameters".
        return {k: float(v) for k, v in data.get("parameters", data).items()
                if isinstance(v, (int, float))}

    import h5py
    with h5py.File(path, "r") as f:
        if "injection_parameters" not in f:
            raise KeyError(f"{path} has no 'injection_parameters' group "
                           "(a real-event file carries no truth)")
        grp = f["injection_parameters"]
        return {k: float(grp[k][()]) for k in grp.keys()}


def load_second_truth(path):
    """Read the second injection at its effective, post-shift merger time."""
    import h5py
    with h5py.File(path, "r") as f:
        if "injection_parameters_2" not in f:
            return None
        grp = f["injection_parameters_2"]
        truth = {k: float(grp[k][()]) for k in grp.keys()}
        if "geocent_time" in truth and "shift_seconds" in grp.attrs:
            truth["geocent_time"] -= float(grp.attrs["shift_seconds"])
        return truth


def get_t_ref(truth_path, times):
    if truth_path is not None and not truth_path.endswith(".json"):
        try:
            from dingo.gw.result import Result
            return float(Result(file_name=truth_path).t_ref)
        except Exception as e:                                   # noqa: BLE001
            print(f"  ! could not read t_ref from {truth_path} ({e})")
    if times is None:
        return DEFAULT_T_REF
    median = float(np.median(times))
    if abs(median - DEFAULT_T_REF) < 10.0:
        return DEFAULT_T_REF
    return float(np.floor(median))


def default_parameters(result):
    """Sampled parameters of the run, in PARAM_ORDER."""
    sampled = [p for p in result.search_parameter_keys if p not in NON_PHYSICAL]
    ordered = [p for p in PARAM_ORDER if p in sampled]
    return ordered + [p for p in sampled if p not in ordered]


def label_of(param, result):
    if param in LATEX:
        return LATEX[param]
    prior = result.priors.get(param) if result.priors is not None else None
    return getattr(prior, "latex_label", None) or param


def density_curve(x, rng, npoints=400, bw=1.0):
    grid = np.linspace(rng[0], rng[1], npoints)
    kde = gaussian_kde(x)
    kde.set_bandwidth(kde.factor * bw)
    return grid, kde(grid)


def is_effectively_constant(values):
    """Whether a sample has too little spread for a stable KDE.

    ``np.std`` is not reliable for a constant array: rounding during its
    reduction can leave a tiny nonzero value (as it does for some Bilby JSON
    posteriors).  Comparing its span against floating-point resolution avoids
    passing that singular data to ``gaussian_kde``.
    """
    values = np.asarray(values)
    scale = max(1.0, float(np.max(np.abs(values))))
    return np.ptp(values) <= 32 * np.finfo(values.dtype).eps * scale


def density_2d(x, y, x_range, y_range, bw=1.0, npoints=128):
    """2D Gaussian KDE on a grid, as (X, Y, Z).

    The 2D counterpart of density_curve: the contours then come from the same
    smooth density as the diagonal curves instead of from a binned histogram,
    so the two halves of the figure cannot disagree about where the posterior is.
    """
    from scipy.stats import gaussian_kde

    kde = gaussian_kde(np.vstack([x, y]))
    kde.set_bandwidth(kde.factor * bw)
    gx = np.linspace(x_range[0], x_range[1], npoints)
    gy = np.linspace(y_range[0], y_range[1], npoints)
    X, Y = np.meshgrid(gx, gy)
    Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
    return X, Y, Z


def credible_levels(Z, levels):
    """Density values enclosing each credible fraction in `levels`.

    Sort the grid by density and walk down until the enclosed mass reaches the
    target: the contour at that height is the smallest region holding it.
    """
    flat = np.sort(Z.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    heights = []
    for level in levels:
        index = np.searchsorted(cumulative, level)
        heights.append(flat[min(index, len(flat) - 1)])
    # contour() needs strictly increasing levels.
    return sorted(set(float(h) for h in heights))


def print_summary(name, df, params, truth_vals):
    print(f"\n{name}: {len(df)} posterior samples")
    print(f"{'parameter':>22} {'median':>12} {'-90% CI':>12} {'+90% CI':>12} {'truth':>12}")
    for i, p in enumerate(params):
        lo, med, hi = np.percentile(df[p].to_numpy(), [5, 50, 95])
        truth_value = None if truth_vals is None else truth_vals[i]
        truth_text = (f"{truth_value:.4f}"
                      if truth_value is not None else "--")
        print(f"{p:>22} {med:>12.4f} {lo:>12.4f} {hi:>12.4f} {truth_text:>12}")


def plot_marginals(frames, params, labels, axis_labels, ranges, truth_vals, bw, out,
                   second_truth_vals=None, selected_signal=None):
    """Write a standalone row of 1D posterior marginal KDEs."""
    fig, axes = plt.subplots(1, len(params), figsize=(4.5 * len(params), 4.0),
                             squeeze=False)
    for i, (param, axis_label, rng) in enumerate(zip(params, axis_labels, ranges)):
        ax = axes[0, i]
        for frame_index, (frame, label) in enumerate(zip(frames, labels)):
            values = frame[param].to_numpy()
            color = f"C{frame_index}"
            if not is_effectively_constant(values):
                grid, density = density_curve(values, rng, bw=bw)
                line, = ax.plot(grid, density, lw=1.5, label=label, color=color)
                lo, med, hi = np.percentile(values, [5, 50, 95])
                # Use the KDE's automatically selected pyplot colour for its
                # 5%, 50%, and 95% quantile markers.
                for quantile in (lo, med, hi):
                    ax.axvline(quantile, color=line.get_color(), ls="--", lw=1.3)
                if len(frames) == 1:
                    latex_name = axis_label.strip("$")
                    ax.set_title(
                        rf"${latex_name} = {med:.2f}^{{+{hi - med:.2f}}}_{{-{med - lo:.2f}}}$"
                    )
            else:
                # Some Bilby runs contain fixed parameters.  A Gaussian KDE
                # has a singular covariance in that case, so show the delta
                # posterior as a vertical line instead.
                ax.axvline(values[0], color=color, lw=1.5, label=label)
        if truth_vals is not None and truth_vals[i] is not None:
            ax.axvline(truth_vals[i], color=("crimson" if selected_signal else "royalblue"),
                       ls=("-" if selected_signal else "--"),
                       label=(f"signal {selected_signal} truth" if selected_signal
                              else "signal 1 truth"))
        if second_truth_vals is not None and second_truth_vals[i] is not None:
            ax.axvline(second_truth_vals[i], color="crimson", ls="-",
                       label="signal 2 truth")
        ax.set_xlim(*rng)
        ax.set_xlabel(axis_label)
        ax.set_ylabel("posterior density" if i == 0 else "")
        if len(frames) > 1 or truth_vals is not None or second_truth_vals is not None:
            ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+",
                    help="bilby *_result.json/hdf5 files, or run directories.")
    ap.add_argument("--labels", default=None,
                    help="Comma-separated legend labels (default: run directory "
                         "names). Only used when several results are given.")
    ap.add_argument("--params", default=None,
                    help="Comma-separated parameters to plot. Default: the "
                         "sampled parameters of the first result.")
    ap.add_argument("--truth", default=None,
                    help="Injection hdf5 (with an injection_parameters group) or "
                         "a truth json, drawn as truth lines.")
    ap.add_argument("--signal", type=int, choices=(1, 2), default=None,
                    help="For a joint Bilby result, plot one signal against its "
                         "own injection truth. Signal 2's geocent_time is derived "
                         "from geocent_time - delta_tc_2.")
    ap.add_argument("--t-ref", type=float, default=None,
                    help=f"Reference GPS time subtracted from geocent_time "
                         f"(default: read from --truth, else {DEFAULT_T_REF}).")
    ap.add_argument("--absolute-time", action="store_true",
                    help="Plot the absolute GPS geocent_time instead of t_c - t_ref.")
    ap.add_argument("--out", default=None,
                    help="Output file (default: <first run dir>/corner.png). "
                        "The extension picks the format.")
    ap.add_argument("--marginal-out", default=None,
                    help="Standalone 1D marginal output (default: <out>_1d.png).")
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--smooth", type=float, default=1.5,
                    help="Gaussian smoothing of the 2D contours, in bins.")
    ap.add_argument("--bw", type=float, default=1.0,
                    help="Bandwidth scale of the KDE curves and contours "
                         "(>1 = smoother).")
    ap.add_argument("--hist-contours", action="store_true",
                    help="Draw corner's binned+smoothed contours in the 2D panels "
                         "instead of KDE contours (the 1D curves stay KDEs).")
    ap.add_argument("--no-titles", action="store_true",
                    help="Do not annotate the 1D panels with median +/- 90%% CI.")
    args = ap.parse_args()

    paths = [resolve_result_path(p) for p in args.results]
    results = []
    for p in paths:
        print(f"loading {p}")
        results.append(load_result(p))

    if args.signal == 2:
        for res in results:
            if "geocent_time_2" not in res.posterior.columns:
                needed = {"geocent_time", "delta_tc_2"}
                if not needed.issubset(res.posterior.columns):
                    sys.exit("signal 2 time requires geocent_time and delta_tc_2")
                res.posterior["geocent_time_2"] = (
                    res.posterior["geocent_time"] - res.posterior["delta_tc_2"]
                )

    params = ([s.strip() for s in args.params.split(",") if s.strip()]
              if args.params else default_parameters(results[0]))
    for res, p in zip(results, paths):
        missing = [q for q in params if q not in res.posterior.columns]
        if missing:
            sys.exit(f"{p}: missing parameter(s) {missing}. Available: "
                     f"{sorted(set(res.posterior.columns) - NON_PHYSICAL)}")
    if len(params) < 2:
        sys.exit("need at least 2 parameters for a corner plot.")

    times = (results[0].posterior["geocent_time"].to_numpy()
             if "geocent_time" in results[0].posterior.columns else None)
    t_ref = (args.t_ref if args.t_ref is not None
             else (getattr(results[0], "t_ref", None)
                   or get_t_ref(args.truth, times)))
    if times is not None and not args.absolute_time:
        print(f"  geocent_time plotted as t_c - {t_ref}")
    truth = load_truth(args.truth) if args.truth else None
    second_truth = (load_second_truth(args.truth)
                    if args.truth and args.truth.endswith((".hdf5", ".h5"))
                    else None)
    if args.signal == 1:
        second_truth = None
    elif args.signal == 2:
        if second_truth is None:
            sys.exit("--signal 2 requires /injection_parameters_2 in --truth")
        truth = {f"{key}_2": value for key, value in second_truth.items()}
        second_truth = None

    frames = []
    for res in results:
        df = res.posterior.copy()
        if not args.absolute_time and not res.time_is_relative:
            for time_key in ("geocent_time", "geocent_time_2"):
                if time_key in df.columns:
                    df[time_key] = df[time_key] - t_ref
        frames.append(df)

    truth_vals = None
    if truth is not None:
        truth_vals = []
        for p in params:
            v = truth.get(p)
            if v is not None and p in ("geocent_time", "geocent_time_2"):
                absolute = abs(v) > 1e6
                if args.absolute_time and not absolute:
                    v = v + t_ref
                elif absolute and not args.absolute_time:
                    v = v - t_ref
            truth_vals.append(v)
        missing = [p for p, v in zip(params, truth_vals) if v is None]
        if missing:
            print(f"  ! truth has no value for {missing}; those panels get no line")

    second_truth_vals = None
    if second_truth is not None:
        second_truth_vals = [second_truth.get(p) for p in params]
        print("  second-signal truth plotted at its effective post-shift time")

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(results):
            sys.exit(f"--labels has {len(labels)} entries for {len(results)} results")
    else:
        labels = [os.path.basename(os.path.dirname(os.path.abspath(p))) for p in paths]

    for name, df in zip(labels, frames):
        print_summary(name, df, params, truth_vals)

    import corner
    axis_labels = [label_of(p, results[0]) for p in params]
    if "geocent_time" in params:
        if args.absolute_time:
            axis_labels[params.index("geocent_time")] = r"$t_c\,[{\rm s}]$"
        elif args.signal:
            axis_labels[params.index("geocent_time")] = (
                r"$t_{c,1}-t_{\mathrm{ref}}\,[{\rm s}]$")
        elif t_ref != DEFAULT_T_REF:
            axis_labels[params.index("geocent_time")] = (
                rf"$t_c - {t_ref:.0f}\,[{{\rm s}}]$")
    if "geocent_time_2" in params:
        axis_labels[params.index("geocent_time_2")] = (
            r"$t_{c,2}\,[{\rm s}]$" if args.absolute_time
            else r"$t_{c,2}-t_{\mathrm{ref}}\,[{\rm s}]$")

    # plot_ranges = {
    # "chirp_mass": (1.18, 1.22),
    # "mass_ratio": (0.70, 1.00),
    # "mass_1": (1.2, 1.8),
    # "mass_2": (1.2, 1.8),
    # "luminosity_distance": (0, 100),
    # "ra": (0, 2 * np.pi),
    # "dec": (-np.pi / 2, np.pi / 2),
    # "geocent_time": (-0.05, 0.05),
    # }

    arrays = [df[params].to_numpy() for df in frames]
    # ranges = [plot_ranges.get(
    #     p,
    #     (min(a[:, i].min() for a in arrays),
    #     max(a[:, i].max() for a in arrays)),
    # )
    # for i, p in enumerate(params)
    # ]
    ranges = []
    for i in range(len(params)):
        values = [min(a[:, i].min() for a in arrays),
                  max(a[:, i].max() for a in arrays)]
        for truths in (truth_vals, second_truth_vals):
            if truths is not None and truths[i] is not None:
                values.append(truths[i])
        lo, hi = min(values), max(values)
        pad = 0.025 * (hi - lo) if hi > lo else 1.0
        ranges.append((lo - pad, hi + pad))
    kde_contours = not args.hist_contours
    fig = None
    for arr_index, (arr, name) in enumerate(zip(arrays, labels)):
        fig = corner.corner(
            arr, fig=fig, color=f"C{arr_index}", labels=axis_labels,
            range=ranges, bins=args.bins,
            smooth=args.smooth,
            plot_datapoints=False, plot_density=False, fill_contours=False,
            # Both the diagonal histogram and the binned contours are drawn
            # invisible (or not at all) and replaced by KDEs below; corner still
            # lays the figure out, and writes the labels, titles and truths.
            plot_contours=not kde_contours,
            levels=(0.5, 0.9), quantiles=[0.05, 0.5, 0.95],
            show_titles=(not args.no_titles and len(arrays) == 1),
            title_kwargs=dict(fontsize=11),
            label_kwargs=dict(fontsize=14),
            hist_kwargs=dict(density=True, lw=0.0, alpha=0.0),
            truths=truth_vals,
            truth_color=("crimson" if args.signal else "royalblue"),
        )

    if second_truth_vals is not None:
        from matplotlib.lines import Line2D
        second_plot_vals = [np.nan if value is None else value
                            for value in second_truth_vals]
        corner.overplot_lines(fig, second_plot_vals, color="crimson",
                              linestyle="-", linewidth=1.6)
        corner.overplot_points(fig, np.asarray(second_plot_vals)[None, :],
                               color="crimson", marker="s", markersize=4)
        handles = [
            Line2D([], [], color="royalblue", lw=1.6, label="signal 1 truth"),
            Line2D([], [], color="crimson", lw=1.6, ls="-", label="signal 2 truth"),
        ]
        fig.legend(handles=handles, loc="upper right",
                   bbox_to_anchor=(0.98, 0.84), fontsize=12, frameon=False)
    elif args.signal and truth_vals is not None:
        from matplotlib.lines import Line2D
        fig.legend(handles=[
            Line2D([], [], color="C0", lw=1.6, label="Bilby posterior"),
            Line2D([], [], color="crimson", lw=1.6,
                   label=f"signal {args.signal} injection truth"),
        ], loc="upper right", bbox_to_anchor=(0.98, 0.84),
            fontsize=12, frameon=False)

    n_par = len(params)
    colors = []
    for j in range(n_par):
        ax = fig.axes[j * (n_par + 1)]
        ymax = 0.0
        for arr_index, arr in enumerate(arrays):
            values = arr[:, j]
            color = f"C{arr_index}"
            if is_effectively_constant(values):
                ax.axvline(values[0], color=color, lw=1.5)
                ymax = max(ymax, 1.0)
                if j == 0:
                    colors.append(color)
                continue
            grid, dens = density_curve(values, ranges[j], bw=args.bw)
            line, = ax.plot(grid, dens, lw=1.5, color=color)
            if j == 0:
                # The contours below have to follow the same order so a result
                # keeps one colour everywhere.
                colors.append(line.get_color())
            ymax = max(ymax, float(dens.max()))
        ax.set_xlim(*ranges[j])
        ax.set_ylim(0.0, 1.1 * ymax)

    if kde_contours:
        # Off-diagonal panel (row j, column i) shows params[i] against params[j].
        for j in range(1, n_par):
            for i in range(j):
                ax = fig.axes[j * n_par + i]
                for arr, color in zip(arrays, colors):
                    x, y = arr[:, i], arr[:, j]
                    if (is_effectively_constant(x) or is_effectively_constant(y)
                            or np.linalg.matrix_rank(np.cov(x, y)) < 2):
                        continue
                    X, Y, Z = density_2d(x, y, ranges[i], ranges[j], bw=args.bw)
                    ax.contour(X, Y, Z, levels=credible_levels(Z, (0.5, 0.9)),
                               colors=color, linewidths=1.2)
                ax.set_xlim(*ranges[i])
                ax.set_ylim(*ranges[j])

    if len(arrays) > 1:
        for name in labels:
            fig.axes[0].plot([], [], label=name)
        if truth_vals is not None and second_truth_vals is None:
            fig.axes[0].plot([], [], ls="-", label="truth")
        fig.legend(loc="upper right", fontsize=16, frameon=False)

    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(paths[0])), "corner.png")
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved corner plot -> {out}")

    out_stem, out_ext = os.path.splitext(out)
    marginal_out = args.marginal_out or f"{out_stem}_1d{out_ext or '.png'}"
    plot_marginals(frames, params, labels, axis_labels, ranges, truth_vals,
                   args.bw, marginal_out, second_truth_vals=second_truth_vals,
                   selected_signal=args.signal)
    print(f"Saved 1D marginal plot -> {marginal_out}")


if __name__ == "__main__":
    main()
