"""Coverage heat maps: one panel per parameter (or per sweep), one shared scale.

  python plot_heatmap.py results/snr_ratio/coverage_snr_ratio.npz \
      --parameters chirp_mass mass_ratio luminosity_distance geocent_time

  python plot_heatmap.py results/*/coverage_*.npz --parameters chirp_mass

The colour is the fraction of a cell's injections whose true signal-1 parameter
landed inside the credible interval: red = 0, green = 1, on a fixed 0-1 scale so
every panel and every figure in the study is measured with the same ruler.

Two things the colour bar alone would hide, and which are therefore drawn:

  * The nominal value is the credible level (0.90 by default), NOT 1.0. Green
    means the posterior is too WIDE, which is a failure in the opposite
    direction from red. The colour bar carries a marker at the nominal value.
  * A cell that lost runs must not look confident. Cells are annotated with
    their own fraction, cells short of the full complement are hatched, and
    empty cells are left blank rather than coloured.
"""

import argparse
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import combo_common as cc

PARAMETER_LABELS = {
    "chirp_mass": r"$\mathcal{M}$",
    "mass_ratio": r"$q$",
    "luminosity_distance": r"$d_L$",
    "geocent_time": r"$t_c$",
    "ra": r"$\alpha$",
    "dec": r"$\delta$",
    "psi": r"$\psi$",
    "mean": "mean over inferred parameters",
}


def _load(path):
    data = np.load(path, allow_pickle=True)
    return {
        "coverage": data["coverage"],
        "n_covered": data["n_covered"],
        "n_total": data["n_total"],
        "parameters": list(data["parameters"]),
        "x_edges": data["x_edges"],
        "y_values": data["y_values"],
        "x_axis": str(data["x_axis"]),
        "x_label": str(data["x_label"]),
        "sweep": str(data["sweep"]),
        "level": float(data["level"]),
        "num_realizations": int(data["num_realizations"]),
        "snr1": float(data["snr1"]),
        "theta1": dict(data["theta1"][0]),
        "baseline": dict(data["baseline"][0]) if "baseline" in data else {},
        "derived": bool(data["binned_by_derived"]) if "binned_by_derived" in data
                   else False,
        "mean_over": [str(p) for p in data["mean_over"]] if "mean_over" in data
                     else [],
        "path": path,
    }


def _y_edges(y_values):
    """Geometric cell boundaries for the log-spaced time-shift axis."""
    y = np.asarray(y_values, dtype=float)
    mid = np.sqrt(y[:-1] * y[1:])
    return np.concatenate([[y[0] ** 2 / mid[0]], mid, [y[-1] ** 2 / mid[-1]]])


def _text_color(cmap, value):
    """Ink that stays legible on the cell underneath it.

    Both ends of a red-yellow-green ramp are dark and the middle is pale, so a
    single fixed ink colour is unreadable somewhere on every figure.
    """
    r, g, b, _ = cmap(value)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if luminance < 0.5 else "#1a1a1a"


def _draw_panel(ax, grid, index, cmap, annotate, num_realizations, show_counts):
    coverage = grid["coverage"][index]
    n_total = grid["n_total"][index]
    true_edges, y_edges = grid["x_edges"], _y_edges(grid["y_values"])

    # A derived axis is binned by equal COUNT, not equal width, so its cells span
    # wildly different ranges (the first sky-separation bin is ~40 deg wide, the
    # middle ones ~9). Drawn at their true positions the narrow cells collapse
    # into unreadable slivers, so derived axes get equal-width cells with the
    # real edges printed on the ticks.
    x_edges = (np.arange(len(true_edges), dtype=float) if grid["derived"]
               else true_edges)
    short_threshold = cc.short_cell_threshold(n_total, num_realizations,
                                              grid["derived"])

    masked = np.ma.masked_invalid(coverage)
    mesh = ax.pcolormesh(x_edges, y_edges, masked, cmap=cmap, vmin=0.0, vmax=1.0,
                         edgecolors="white", linewidth=0.5)

    ax.set_yscale("log")
    ax.set_yticks(grid["y_values"])
    ax.set_yticklabels([f"{v:.2f}" for v in grid["y_values"]], fontsize=7)
    ax.minorticks_off()
    if (not grid["derived"] and np.all(x_edges > 0)
            and x_edges[-1] / max(x_edges[0], 1e-12) > 20):
        ax.set_xscale("log")
    ax.tick_params(axis="x", labelsize=7)
    for spine in ax.spines.values():
        spine.set_visible(False)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    if ax.get_xscale() == "log":
        x_centers = np.sqrt(x_edges[:-1] * x_edges[1:])
    y_centers = grid["y_values"]

    # Tick every bin, like the y axis. The grid is discrete, so decade ticks
    # would sit between cells and invite reading the axis as continuous. On a
    # derived axis the ticks go on the EDGES, since the cells are equal-width on
    # screen and the edges are the only honest statement of what they span.
    if grid["derived"]:
        ax.set_xticks(x_edges)
        ax.set_xticklabels([f"{v:.3g}" for v in true_edges], fontsize=6,
                           rotation=45, ha="right")
    else:
        ax.set_xticks(x_centers)
        ax.set_xticklabels([f"{v:.3g}" for v in x_centers], fontsize=6,
                           rotation=45, ha="right")
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    for j in range(coverage.shape[0]):
        for i in range(coverage.shape[1]):
            if n_total[j, i] == 0:
                # Nothing ran here: leave the surface bare so it cannot be read
                # as a measurement.
                ax.add_patch(Rectangle(
                    (x_edges[i], y_edges[j]),
                    x_edges[i + 1] - x_edges[i], y_edges[j + 1] - y_edges[j],
                    facecolor="#f0f0f0", edgecolor="white", linewidth=0.5, zorder=1.5))
                continue
            if n_total[j, i] < short_threshold:
                ax.add_patch(Rectangle(
                    (x_edges[i], y_edges[j]),
                    x_edges[i + 1] - x_edges[i], y_edges[j + 1] - y_edges[j],
                    facecolor="none", edgecolor="#404040", hatch="///",
                    linewidth=0.0, zorder=2))
            if annotate == "none":
                continue
            value = coverage[j, i]
            text = f"{value:.2f}"
            if show_counts:
                text += f"\n{int(grid['n_covered'][index][j, i])}/{int(n_total[j, i])}"
            ax.text(x_centers[i], y_centers[j], text, ha="center", va="center",
                    fontsize=5.5 if show_counts else 6.5,
                    color=_text_color(cmap, value), zorder=3)
    return mesh


def _header(fig, grids, level, parameter, chrome=True, by_parameter=False):
    """Title, and -- only with `chrome` -- the subtitle and the fixed/varied box.

    Positioned in inches converted to figure fractions, not in fractions
    directly: the figure grows with the panel count, and fixed fractions would
    let the subtitle ride up into the title on a one-row figure.
    """
    height = fig.get_size_inches()[1]
    def _y(inches_from_top):
        return 1.0 - inches_from_top / height

    g = grids[0]
    theta1 = g["theta1"]
    fixed = (f"$\\mathcal{{M}}_1$ = {theta1['chirp_mass']:.2f} $M_\\odot$   |   "
             f"$q_1$ = {theta1['mass_ratio']:.3f}   |   "
             f"$d_{{L,1}}$ = {theta1['luminosity_distance']:.0f} Mpc   |   "
             f"SNR$_1$ = {g['snr1']:.1f}")
    varied = "   |   ".join(["$|\\Delta t_c|$"] + sorted({gg["x_label"] for gg in grids}))

    # The mean is an average of per-parameter coverages, not a quantity whose
    # own truth falls inside an interval -- it needs its own wording.
    if by_parameter and len(grids) > 1:
        title = "Target recovery against a distraction BBH signal"
        subtitle = ("Fixed target signal. Columns are the recovered parameters, "
                    "rows the varied property of the distraction signal; colour "
                    f"= fraction of injections covering the truth at {level:.0%}.")
    elif by_parameter:
        # Panels are the parameters, so the figure names the x axis instead.
        title = f"Target recovery vs {g['x_label']}"
        subtitle = (f"Fixed target signal. Each panel is one recovered parameter; "
                    f"colour = fraction of injections whose true value fell inside "
                    f"the {level:.0%} credible interval.")
    elif parameter == "mean":
        members = g.get("mean_over") or []
        title = "Target recovery against a distraction BBH signal"
        subtitle = (f"Fixed target signal; only the distraction signal is varied. "
                    f"Colour = {level:.0%} coverage of the target's true "
                    f"parameters, averaged over "
                    f"{f'the {len(members)} inferred parameters' if members else 'the inferred parameters'}.")
    else:
        scored = PARAMETER_LABELS.get(parameter, parameter)
        title = f"Recovery of the target's {scored} against a distraction BBH signal"
        subtitle = (f"Fixed target signal; only the distraction signal is varied. "
                    f"Colour = fraction of injections whose true {scored} fell "
                    f"inside the {level:.0%} credible interval.")
    fig.text(0.5, _y(0.30 if chrome else 0.42), title,
             ha="center", va="top", fontsize=15, fontweight="bold", color="#1a1a1a")
    if not chrome:
        return
    fig.text(0.5, _y(0.62), subtitle,
             ha="center", va="top", fontsize=9, color="#555555")

    box = fig.add_axes([0.035, _y(1.42), 0.80, 0.45 / height])
    box.set_xticks([]); box.set_yticks([])
    for side, spine in box.spines.items():
        spine.set_edgecolor("#1f7a6c")
        spine.set_linewidth(1.0)
    box.set_facecolor("#f2f8f6")
    box.text(0.012, 0.72, "Fixed target signal", fontsize=8.5, fontweight="bold",
             color="#1f7a6c", transform=box.transAxes)
    box.text(0.012, 0.24, fixed, fontsize=8, color="#333333", transform=box.transAxes)
    box.text(0.560, 0.72, "Varied distraction signal", fontsize=8.5,
             fontweight="bold", color="#1f7a6c", transform=box.transAxes)
    box.text(0.560, 0.24, varied, fontsize=8, color="#333333", transform=box.transAxes)


def plot(grids, parameter, out_path, cmap_name="RdYlGn", annotate="value",
         show_counts=False, ncols=2, chrome=False, panel_parameters=None):
    """One figure of heat maps sharing a single 0-1 colour scale.

    Two ways to fill the panels, and one rule decides the titles: **a panel is
    titled by whatever changes between panels**, since that is the only thing a
    reader needs the title to disambiguate.

    * `panel_parameters` given -> one panel per scored parameter, all sharing
      the figure's single x axis. Panels are titled by the parameter; the x
      axis is named once, on each panel's x label.
    * otherwise -> one panel per input grid (per x axis), all scoring the same
      parameter. Panels are titled by their x axis, and the figure names the
      parameter.
    """
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad("#f0f0f0")

    by_parameter = bool(panel_parameters)
    if by_parameter:
        # One row per input grid, one column per parameter. With a single grid
        # this is just a row of panels; with several it is the matrix, where the
        # x axis changes down the rows and the scored parameter across the
        # columns. Only the top row is titled -- repeating the parameter above
        # every row would be noise, and the x axis is named on each row's own
        # x label anyway.
        panels, titles = [], []
        for row, grid in enumerate(grids):
            missing = [p for p in panel_parameters if p not in grid["parameters"]]
            if missing:
                print(f"   (not in {os.path.basename(grid['path'])}: {missing})")
            for p in panel_parameters:
                if p not in grid["parameters"]:
                    continue
                panels.append((grid, grid["parameters"].index(p)))
                titles.append(PARAMETER_LABELS.get(p, p) if row == 0 else "")
    else:
        panels = [(g, g["parameters"].index(parameter))
                  for g in grids if parameter in g["parameters"]]
        titles = [g["x_label"] for g, _ in panels]
    if not panels:
        raise SystemExit(f"Nothing to plot from the given files.")

    if by_parameter and len(grids) > 1:
        ncols = len([p for p in panel_parameters
                     if p in grids[0]["parameters"]])   # one column per parameter
    ncols = min(ncols, len(panels))
    nrows = int(np.ceil(len(panels) / ncols))
    # A single panel still needs a wide canvas: the header runs two columns and
    # the footnote two long lines, and both overflow below about 10.5 inches.
    # The panel is made taller to stop the 10x10 grid stretching into ribbons.
    if len(panels) == 1:
        panel_w, panel_h = 7.5, 6.0
    elif len(panels) > 6:
        panel_w, panel_h = 4.8, 3.8     # a 4x3 matrix would otherwise be enormous
    else:
        panel_w, panel_h = 5.6, 4.1
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(max(panel_w * ncols, 10.5), panel_h * nrows + 1.9),
        squeeze=False)
    fig.patch.set_facecolor("white")

    level = grids[0]["level"]
    num_realizations = grids[0]["num_realizations"]
    mesh = None
    for panel_index, (grid, index) in enumerate(panels):
        ax = axes[panel_index // ncols][panel_index % ncols]
        mesh = _draw_panel(ax, grid, index, cmap, annotate, num_realizations,
                           show_counts)
        # Titled by whatever differs between panels -- the parameter when the x
        # axis is shared, the x axis when the parameter is shared.
        ax.set_title(titles[panel_index], fontsize=11, fontweight="bold",
                     color="#1a1a1a", pad=14 if chrome else 6)
        # The comparison value is the control, NOT the credible level. One fixed
        # target signal cannot give nominal coverage even for a perfect sampler,
        # so a panel read against 0.90 would be read wrong.
        base = grid["baseline"].get(
            grid["parameters"][index] if by_parameter else parameter)
        if base is not None and chrome:
            ax.text(0.5, 1.012, f"no-intruder baseline {base:.2f}", fontsize=7.5,
                    color="#777777", ha="center", va="bottom",
                    transform=ax.transAxes)
        ax.set_xlabel(grid["x_label"], fontsize=9, color="#333333")
        ax.set_ylabel(r"$|\Delta t_c|$  [s]", fontsize=9, color="#333333")
    for spare in range(len(panels), nrows * ncols):
        axes[spare // ncols][spare % ncols].axis("off")

    height = fig.get_size_inches()[1]
    # Without the chrome the plot reclaims the caption strip at the bottom and
    # most of the header strip at the top.
    fig.tight_layout(rect=[0.0, (0.54 if chrome else 0.10) / height, 0.88,
                           1.0 - (1.65 if chrome else 0.75) / height])
    _header(fig, grids, level, parameter, chrome=chrome,
            by_parameter=by_parameter)

    cax = fig.add_axes([0.905, 0.12, 0.018, 0.60])
    bar = fig.colorbar(mesh, cax=cax)
    bar.set_label(f"{level:.0%} coverage", fontsize=9, color="#333333")
    bar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    bar.ax.set_yticklabels(["0.0", "0.25", "0.5", "0.75", "1.0"], fontsize=8)
    # No "nominal" marker on the bar: the value a cell should be compared with
    # is the per-parameter control printed under each panel title, not the
    # credible level, and the two are far apart (0.77 vs 0.90 for chirp_mass).

    sigma = np.sqrt(level * (1 - level) / max(num_realizations, 1))
    # Two lines: as one line this runs off the canvas on a 2-column figure.
    # Caption lines are stacked from the bottom in fixed inch steps; the
    # "averaged over" line only exists for the mean, so everything above it
    # shifts up by one step when it is present.
    members = (grids[0]["mean_over"]
               if (parameter == "mean" and not by_parameter) else [])
    step = 0.12
    base = 0.08 + (step if members else 0.0)
    if not chrome:
        _finish(fig, out_path)
        return out_path
    fig.text(0.5, (base + step) / height,
             f"Compare each cell with ITS PANEL'S no-intruder baseline, not with "
             f"{level:.0%} — one fixed target signal cannot give nominal coverage "
             f"even for a perfect sampler.",
             ha="center", fontsize=7.5, color="#777777")
    fig.text(0.5, base / height,
             f"{num_realizations} injections per cell, binomial uncertainty "
             f"~{sigma:.02f}.   Hatched = incomplete cell, grey = no runs.   "
             f"Signal 2 merges $|\\Delta t_c|$ earlier than signal 1.",
             ha="center", fontsize=7.5, color="#777777")
    if members:
        # Naming the members matters: psi and mass_ratio have prior-wide
        # posteriors that sit at 1.00 coverage, so including them puts a floor
        # under every cell that a reader has to know about. Its own line --
        # appended to the line above it runs off the canvas.
        fig.text(0.5, 0.08 / height,
                 "Averaged over: " + ", ".join(members).replace("_", " ") + ".",
                 ha="center", fontsize=7.5, color="#777777")

    _finish(fig, out_path)
    return out_path


def _finish(fig, out_path):
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=180, facecolor="white")
    plt.close(fig)
    print(f"-> {cc.rel(os.path.abspath(out_path))}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("npz", nargs="+", help="coverage_*.npz from summarize_grid.py")
    p.add_argument("--parameters", nargs="+", default=["chirp_mass", "mass_ratio",
                                                       "luminosity_distance",
                                                       "geocent_time"])
    p.add_argument("--out", default=None)
    p.add_argument("--cmap", default="RdYlGn",
                   help="RdYlGn is the requested red-to-green scale. Use "
                        "'cividis' or 'viridis' for a colour-vision-safe "
                        "version of the same figure -- the in-cell numbers "
                        "carry the value either way.")
    p.add_argument("--annotate", default="value", choices=("value", "none"))
    p.add_argument("--counts", action="store_true",
                   help="Also print n_covered/n_total inside each cell.")
    p.add_argument("--ncols", type=int, default=2)
    p.add_argument("--panels", default="axis", choices=("axis", "parameter", "matrix"),
                   help="'axis' (default): one figure per parameter, one panel "
                        "per input file. 'parameter': one figure per input file, "
                        "one panel per --parameters entry, all sharing that "
                        "file's x axis. 'matrix': ONE figure, rows = input "
                        "files, columns = --parameters.")
    p.add_argument("--chrome", action="store_true",
                   help="Restore the subtitle, the fixed/varied box and the "
                        "caption. Off by default: the figure is the heat map "
                        "and its title. The context those lines carried -- the "
                        "no-intruder baseline, the per-cell count, which "
                        "parameters the mean runs over -- is still in the .npz "
                        "and printed by summarize_grid.py.")
    args = p.parse_args(argv)

    grids = [_load(path) for path in args.npz]
    axes_stem = "_".join(sorted({g["x_axis"] for g in grids}))
    # The sweep has to be in the name. Two figures can share an x axis and a
    # scored parameter and still be different measurements -- the controlled
    # snr_ratio sweep and the population sweep re-binned by snr_ratio are the
    # same axes over completely differently generated injections.
    sweep_stem = "_".join(sorted({g["sweep"] for g in grids}))

    # One figure per parameter: a single figure mixing them would have to title
    # its panels by the parameter, which is exactly what makes the x axis and
    # the panel title disagree.
    outputs = []
    if args.panels == "matrix":
        # ONE figure: rows = input files (x axes), columns = parameters.
        stem = "_".join(args.parameters)
        out = args.out or os.path.join(
            cc.RESULTS_DIR, "figures", f"matrix__{stem}_vs_{sweep_stem}.png")
        return [plot(grids, None, out, cmap_name=args.cmap,
                     annotate=args.annotate, show_counts=args.counts,
                     ncols=args.ncols, chrome=args.chrome,
                     panel_parameters=args.parameters)]

    if args.panels == "parameter":
        # One figure per input file; the panels are the parameters.
        for grid in grids:
            stem = "_".join(args.parameters)
            out = args.out or os.path.join(
                cc.RESULTS_DIR, "figures",
                f"{grid['sweep']}__{stem}_vs_{grid['x_axis']}.png")
            outputs.append(plot([grid], None, out, cmap_name=args.cmap,
                                annotate=args.annotate, show_counts=args.counts,
                                ncols=args.ncols, chrome=args.chrome,
                                panel_parameters=args.parameters))
            if args.out:
                break
        return outputs

    for parameter in args.parameters:
        if not any(parameter in g["parameters"] for g in grids):
            print(f"   (skipping {parameter}: not in these files)")
            continue
        out = args.out or os.path.join(
            cc.RESULTS_DIR, "figures",
            f"{sweep_stem}__{parameter}_vs_{axes_stem}.png")
        outputs.append(plot(grids, parameter, out, cmap_name=args.cmap,
                            annotate=args.annotate, show_counts=args.counts,
                            ncols=args.ncols, chrome=args.chrome))
        if args.out:  # an explicit path can only receive one figure
            break
    return outputs


if __name__ == "__main__":
    main()
