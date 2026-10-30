#!/usr/bin/env python
"""Jensen-Shannon divergence between Dingo posteriors and their bilby baseline.

Ported from bilby_baseline/compare.py. The divergence itself is unchanged; what
moved is everything around it -- the script now addresses posteriors the way the
rest of analysis/ does, by target name and model, reusing the discovery and the
loaders in plot_posteriors.py instead of carrying its own copies. That removes
the failure this script used to have: its private load_dingo() rebuilt the frame
as pd.DataFrame(np.array(result.samples)), which drops the column names, so every
parameter silently failed to match and the table came out empty.

Units are nats (base e), so the upper bound is ln 2 = 0.693, not 1. Divide by
ln 2 for bits. The value written to js_divergence.txt carries its unit in the
header, because a table of bare numbers is unreadable six months later.

A JS divergence between finite sample sets is biased upward -- with a few
hundred samples per posterior that bias is not small. Run js_noise_floor.py on
the same pair to measure it before reading anything into a value.

Examples
--------
    # a target with one model
    python compare.py seed0053_dprior --model ET_best

    # several models against the same baseline, one column each
    python compare.py seed0053_dprior --model ET_best ET_latest

    # explicit paths, bypassing discovery
    python compare.py --dingo run/result.hdf5 --bilby run/bilby_result.json

Outputs, under tables/<target>/ unless --outdir says otherwise:

    js_divergence.txt           the table, tab separated, unit in the header
    dingo_vs_bilby_corner.pdf   overlaid corner

The corner here is the quick look that ships with the table; plot_posteriors.py
draws the better one (KDE contours, injected truth, consistent styling).
"""
import argparse
import os

import numpy as np
from scipy.spatial.distance import jensenshannon

# analysis/ is meant to be run from analysis/, so a plain import is enough. The
# loaders and the naming conventions live in plot_posteriors so that there is
# one definition of where a posterior lives and how bilby's time convention is
# reconciled with Dingo's.
from plot_posteriors import (
    COLORS,
    RESULT_ROOT,
    _model_tag,
    _rel,
    _target_of,
    align_time_convention,
    find_bilby,
    find_dingo_results,
    load_bilby,
    load_dingo,
)

TABLES_DIR = os.path.join(RESULT_ROOT, "tables")

# Compared wherever both posteriors provide them. geocent_time is appended
# separately: it is the one parameter whose convention differs between the two
# codes, so it only joins the list after align_time_convention has run.
COMMON_PARAMS = [
    "chirp_mass", "mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2", "chi_eff", "chi_p",
    "theta_jn", "luminosity_distance", "ra", "dec", "psi", "phi_12", "phi_jl",
]

LATEX = {
    "chirp_mass": r"$\mathcal{M}$", "mass_ratio": r"$q$",
    "a_1": r"$a_1$", "a_2": r"$a_2$", "tilt_1": r"$\theta_1$",
    "tilt_2": r"$\theta_2$", "theta_jn": r"$\theta_{JN}$",
    "luminosity_distance": r"$d_L$", "ra": r"$\alpha$", "dec": r"$\delta$",
    "psi": r"$\psi$", "phi_12": r"$\phi_{12}$", "phi_jl": r"$\phi_{JL}$",
    "chi_eff": r"$\chi_{\rm eff}$", "chi_p": r"$\chi_p$",
    "geocent_time": r"$t_c - t_{\rm ref}$",
}

DEFAULT_BINS = 40

# Equal-tailed credible interval, matching plot_posteriors.print_summary, so the
# width here and the "90% CI" printed there are the same quantity.
DEFAULT_CREDIBLE = 0.90


def js_divergence_nat(x, y, bins=DEFAULT_BINS, weights_x=None, weights_y=None):
    """1D Jensen-Shannon divergence in nats between two sample sets.

    Both sets are binned on a common support so the two histograms are
    comparable. scipy returns the JS *distance* (the square root of the
    divergence), hence the square. Empty bins need no epsilon: scipy's rel_entr
    defines 0 * log(0/m) as 0.
    """
    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    edges = np.linspace(lo, hi, bins + 1)

    p, _ = np.histogram(x, bins=edges, weights=weights_x)
    q, _ = np.histogram(y, bins=edges, weights=weights_y)

    p = p / p.sum()
    q = q / q.sum()

    d = jensenshannon(p, q, base=np.e)
    return float(d ** 2)


def weighted_quantile(x, q, weights=None):
    """Quantiles of `x` at levels `q`, honouring sample weights.

    np.percentile cannot weight, and a Dingo result that has been importance
    sampled carries a weights column -- ignoring it there would measure the
    width of the proposal rather than of the posterior.
    """
    x = np.asarray(x, dtype=float)
    q = np.atleast_1d(np.asarray(q, dtype=float))
    if weights is None:
        return np.percentile(x, 100.0 * q)
    w = np.asarray(weights, dtype=float)
    order = np.argsort(x)
    x, w = x[order], w[order]
    # Midpoint convention: the cumulative weight at each sample is taken at the
    # centre of that sample's weight, which reduces to np.percentile's linear
    # interpolation when all weights are equal.
    cw = np.cumsum(w)
    cw = (cw - 0.5 * w) / cw[-1]
    return np.interp(q, cw, x)


def credible_width(x, weights=None, credible=DEFAULT_CREDIBLE):
    """Width of the equal-tailed credible interval, W_<credible>."""
    tail = 0.5 * (1.0 - credible)
    lo, hi = weighted_quantile(x, [tail, 1.0 - tail], weights)
    return float(hi - lo)


def common_parameters(frames):
    """The compared parameters every one of `frames` provides, in a fixed order."""
    params = [p for p in COMMON_PARAMS if all(p in f.columns for f in frames)]
    if all("geocent_time" in f.columns for f in frames):
        params.append("geocent_time")
    return params


def write_table(path, stats, bilby_w, params, labels, bins, credible,
                dingo_paths, bilby_path):
    """The comparison table, with enough provenance to be readable on its own."""
    pct = int(round(100 * credible))
    with open(path, "w") as f:
        f.write(f"# JSD  Jensen-Shannon divergence vs bilby, nats (base e); "
                f"upper bound ln2 = {np.log(2):.4f}, {bins} histogram bins\n")
        f.write(f"# W{pct}  width of the equal-tailed {pct}% credible interval, "
                f"in the parameter's own units\n")
        f.write(f"# R{pct}  W{pct}(Dingo) / W{pct}(bilby); 1 = same width, "
                f">1 = Dingo broader, <1 = Dingo overconfident\n")
        f.write(f"# bilby: {_rel(bilby_path)}\n")
        for label, path_i in zip(labels, dingo_paths):
            f.write(f"# dingo [{label}]: {_rel(path_i)}\n")
        columns = [f"bilby_W{pct}"]
        for label in labels:
            columns += [f"{label}_JSD", f"{label}_W{pct}", f"{label}_R{pct}"]
        f.write("parameter\t" + "\t".join(columns) + "\n")
        for p in params:
            row = [f"{bilby_w[p]:.6g}"]
            for label in labels:
                st = stats[label][p]
                row += [f"{st['js']:.6f}", f"{st['w']:.6g}", f"{st['r']:.4f}"]
            f.write(p + "\t" + "\t".join(row) + "\n")


def plot_corner(out, dingo_runs, bilby_df, bilby_w, params):
    """Overlaid corner: every Dingo posterior against the bilby baseline."""
    import corner

    frames = [df for _, df, _ in dingo_runs] + [bilby_df]
    arrays = [f[params].to_numpy() for f in frames]
    ranges = [
        (min(a[:, i].min() for a in arrays), max(a[:, i].max() for a in arrays))
        for i in range(len(params))
    ]

    fig = None
    handles = []
    for i, (label, df, weights) in enumerate(dingo_runs):
        color = COLORS[i % len(COLORS)]
        fig = corner.corner(
            df[params].to_numpy(), fig=fig,
            labels=[LATEX.get(p, p) for p in params],
            color=color, range=ranges, weights=weights,
            plot_datapoints=False, plot_density=False,
            levels=(0.5, 0.9), hist_kwargs=dict(density=True),
        )
        handles.append((color, f"Dingo {label}"))

    bilby_color = COLORS[len(dingo_runs) % len(COLORS)]
    corner.corner(
        bilby_df[params].to_numpy(), fig=fig, color=bilby_color, range=ranges,
        weights=bilby_w, plot_datapoints=False, plot_density=False,
        levels=(0.5, 0.9), hist_kwargs=dict(density=True),
    )
    handles.append((bilby_color, "bilby (baseline)"))

    for color, label in handles:
        fig.axes[0].plot([], [], color=color, label=label)
    fig.legend(loc="upper right", fontsize=14)
    fig.savefig(out, bbox_inches="tight")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=None,
                    help="Event or injection name, e.g. seed0053_dprior. Run "
                         "'python plot_posteriors.py --list' to see them. Omit "
                         "only when passing --dingo.")
    ap.add_argument("--model", nargs="+", default=None,
                    help="Which model's posterior(s) to compare. Several names "
                         "give one table column each. Default: the only model "
                         "that has this target, erroring if there is more than "
                         "one.")
    ap.add_argument("--dingo", nargs="+", default=None,
                    help="Explicit Dingo hdf5 path(s), bypassing discovery.")
    ap.add_argument("--bilby", default=None,
                    help="Explicit bilby result (.json/.hdf5). Default: "
                         "auto-discovered from the target name.")
    ap.add_argument("--outdir", default=None,
                    help=f"Default: {_rel(TABLES_DIR)}/<target>/.")
    ap.add_argument("--bins", type=int, default=DEFAULT_BINS,
                    help=f"Histogram bins per parameter. Default {DEFAULT_BINS}. "
                         f"A real difference is stable against this; sweep it "
                         f"with js_noise_floor.py to confirm.")
    ap.add_argument("--credible", type=float, default=DEFAULT_CREDIBLE,
                    help=f"Credible level for the interval-width ratio. "
                         f"Default {DEFAULT_CREDIBLE} -> R90.")
    args = ap.parse_args()

    if not 0.0 < args.credible < 1.0:
        raise SystemExit("--credible must lie strictly between 0 and 1.")

    if not args.target and not args.dingo:
        ap.print_help()
        print("\nNothing to compare. Start with:  "
              "python plot_posteriors.py --list")
        raise SystemExit(1)

    found = find_dingo_results(args.target, args.model, args.dingo)
    target = args.target or _target_of(found[0][1])

    dingo_runs, dingo_paths, t_ref = [], [], None
    for label, path in found:
        print(f"loading Dingo   [{label}] {_rel(path)}")
        samples, weights, _, _, t_ref_i = load_dingo(path)
        if t_ref is not None and t_ref_i != t_ref:
            raise SystemExit(
                f"Posteriors disagree on the reference time ({t_ref} vs "
                f"{t_ref_i}); they are not the same event/injection.")
        t_ref = t_ref_i
        dingo_runs.append((label, samples, weights))
        dingo_paths.append(path)

    bilby_path = args.bilby or find_bilby(target)
    if not bilby_path or not os.path.exists(bilby_path):
        raise SystemExit(
            f"No bilby baseline for {target}; there is nothing to compare "
            f"against.\nProduce one from bilby_baseline/:\n"
            f"  RESULT={_rel(dingo_paths[0])} sbatch submit_injection_baseline.sh\n"
            f"  EVENT={target} sbatch real_event.sh")
    print(f"loading bilby   {_rel(bilby_path)}")
    bilby_df = align_time_convention(load_bilby(bilby_path), t_ref)
    bilby_w = None

    params = common_parameters([df for _, df, _ in dingo_runs] + [bilby_df])
    if not params:
        raise SystemExit("No parameters in common between the posteriors.")

    outdir = os.path.join(args.outdir or TABLES_DIR, target)
    os.makedirs(outdir, exist_ok=True)

    labels = [label for label, _, _ in dingo_runs]
    pct = int(round(100 * args.credible))

    # The baseline widths, once -- every ratio is taken against these.
    bilby_width = {
        p: credible_width(bilby_df[p].to_numpy(), bilby_w, args.credible)
        for p in params
    }

    stats = {}
    for label, df, weights in dingo_runs:
        per_param = {}
        for p in params:
            width = credible_width(df[p].to_numpy(), weights, args.credible)
            per_param[p] = {
                "js": js_divergence_nat(df[p].to_numpy(),
                                        bilby_df[p].to_numpy(),
                                        bins=args.bins, weights_x=weights,
                                        weights_y=bilby_w),
                "w": width,
                # A zero-width baseline would be a degenerate bilby posterior
                # rather than perfect agreement, so leave it undefined instead
                # of dividing.
                "r": width / bilby_width[p] if bilby_width[p] > 0 else np.nan,
            }
        stats[label] = per_param

    col = max(12, max(len(l) for l in labels) + 2)

    def block(title, key, fmt, summary=True):
        print(f"\n{title}")
        print(f"{'parameter':>20}" + "".join(f"{l:>{col}}" for l in labels))
        for p in params:
            print(f"{p:>20}"
                  + "".join(f"{stats[l][p][key]:>{col}{fmt}}" for l in labels))
        if summary:
            print(f"{'median':>20}"
                  + "".join(
                      f"{np.median([stats[l][p][key] for p in params]):>{col}{fmt}}"
                      for l in labels))

    block(f"JS divergence vs bilby, nats (base e), {args.bins} bins; "
          f"upper bound ln2 = {np.log(2):.4f}", "js", ".4f")

    block(f"R{pct} = W{pct}(Dingo) / W{pct}(bilby)   "
          f"[1 = same width, >1 = Dingo broader]", "r", ".2f")

    for l in labels:
        js_l = {p: stats[l][p]["js"] for p in params}
        r_l = {p: stats[l][p]["r"] for p in params}
        worst_js = max(js_l, key=js_l.get)
        worst_r = max(r_l, key=lambda p: abs(np.log(r_l[p])) if r_l[p] > 0
                      else -np.inf)
        print(f"\n  [{l}] largest JSD: {worst_js} ({js_l[worst_js]:.4f})")
        print(f"  [{l}] furthest R{pct} from 1: {worst_r} "
              f"({r_l[worst_r]:.2f}x)")

    tag = _model_tag(dingo_runs)
    table = os.path.join(outdir, f"{target}_{tag}_comparison.txt")
    write_table(table, stats, bilby_width, params, labels, args.bins,
                args.credible, dingo_paths, bilby_path)
    print(f"\nSaved table       -> {_rel(table)}")

    corner_out = plot_corner(
        os.path.join(outdir, f"{target}_{tag}_dingo_vs_bilby_corner.pdf"),
        dingo_runs, bilby_df, bilby_w, params)
    print(f"Saved corner plot -> {_rel(corner_out)}")


if __name__ == "__main__":
    main()
