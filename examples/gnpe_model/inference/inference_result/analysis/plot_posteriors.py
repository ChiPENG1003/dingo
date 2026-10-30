#!/usr/bin/env python
"""Corner and 1D-marginal figures for Dingo posteriors, with an optional bilby baseline.

Replaces plot_real_event_corner.py (real events only), make_corner.py (one hardcoded
injection, via CSV) and plot_corner_dingo_vs_bilby.py (a thin duplicate). Works on
real events and injections alike, because the only thing that differed between them
was the geocent_time reference -- and Result.t_ref already resolves that.

START HERE
----------
    python plot_posteriors.py --list                    what can be plotted, and what
                                                    already has a bilby baseline

    python plot_posteriors.py GW170809                  one target, auto-discovered
    python plot_posteriors.py seed0053_dprior           injections work the same way

Targets are named the way the posterior files are: a real event is 'GW170809'
(posteriors/<model>/real/GW170809_posterior.hdf5) and an injection is
'seed0053_dprior' (posteriors/<model>/single/seed0053_dprior.hdf5). --list prints
the exact names.

COMMON JOBS
-----------
    # compare two networks on one event, against bilby
    python plot_posteriors.py GW170809 --model chi_eff 1M_7params_O2

    # the ET injection, only the parameters that model infers
    python plot_posteriors.py seed0053_dprior --model ET \
        -p chirp_mass mass_ratio chi_eff luminosity_distance

    # Dingo only, no baseline
    python plot_posteriors.py GW170809 --bilby none

    # explicit files, ignoring the naming convention entirely
    python plot_posteriors.py --dingo a.hdf5 b.hdf5 --bilby run/bilby_result.json

OUTPUT (into inference_result/figures/<target>/, or --outdir/<target>/)
    <target>_<models>_dingo_corner.png      Dingo posterior(s) alone
    <target>_<models>_dingo_vs_bilby.png    the same, overlaid on bilby
    <target>_<models>_dingo_vs_bilby_1d.png 1D marginals of the above

  <models> is the model labels joined with '+' ('ET', 'ET+chi_eff'), so a
  comparison never overwrites a single-model figure of the same target.
plus a median / 90% CI table on stdout.

NOTES
    - An injection's truth is read from the file and drawn as a black cross.
      Real events have none, so nothing is drawn.
    - Models that infer different parameters reduce the figure to their
      intersection, and say so.
    - bilby's absolute-GPS geocent_time is shifted onto Dingo's relative
      convention automatically.
"""
import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde

from dingo.core.utils.plotting import get_latex_labels
from dingo.gw.result import Result

# Credible regions drawn as 2D KDE contours, and the KDE grid resolution.
CREDIBLE_LEVELS = (0.5, 0.9)
GRID = 256  # a bilby posterior can be ~1% the width of a Dingo one; 128 left
            # it spanning only two or three cells
COLORS = ("#1b7f79", "#d1622b", "#4c5bd4", "#9c27b0")
TRUTH_LW = 0.9

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.dirname(HERE)
LEGACY_DIR = os.path.join(RESULT_ROOT, "legacy")
FIGURES_DIR = os.path.join(RESULT_ROOT, "figures")
POSTERIOR_DIR = os.path.join(RESULT_ROOT, "posteriors")

# inference_result/analysis -> inference_result -> inference -> gnpe_model
EXAMPLE_DIR = os.path.dirname(os.path.dirname(RESULT_ROOT))
BILBY_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(EXAMPLE_DIR)), "bilby_baseline")
BILBY_EVENT_DIR = os.path.join(BILBY_ROOT, "bilby_real_events")
BILBY_INJECTION_DIR = os.path.join(BILBY_ROOT, "bilby_injections")

DEFAULT_PARAMETERS = [
    "chirp_mass", "mass_ratio", "chi_eff", "luminosity_distance",
    # "geocent_time", "ra", "dec", "psi",
]

TIME_LABEL = r"$t_c - t_{\rm ref}$ [s]"

# get_latex_labels() only knows parameters that appear in a prior, so derived
# quantities a network may infer would otherwise show as a bare column name.
DERIVED_LATEX = {
    "chi_eff": r"$\chi_{\rm eff}$",
    "chi_p": r"$\chi_p$",
    "chi_1z": r"$\chi_{1z}$",
    "chi_2z": r"$\chi_{2z}$",
}


def _picture_path(name, subdir=None):
    """Path for an output figure, under figures/<subdir>/ when a subdir is given.

    One directory per target keeps the three figures of a run together, so a
    figures/ directory that has accumulated dozens of targets stays navigable.
    """
    directory = os.path.join(FIGURES_DIR, subdir) if subdir else FIGURES_DIR
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, name)


def _model_tag(dingo_runs, max_len=60):
    """Filesystem-safe join of the model labels in a figure.

    'ET', or 'ET+chi_eff' for a comparison. Truncated with a count if a run
    stacks so many models that the name would get unwieldy.
    """
    labels = [re.sub(r"[^0-9A-Za-z._-]+", "-", str(label)) for label, _, _ in dingo_runs]
    tag = "+".join(labels)
    if len(tag) > max_len:
        tag = f"{labels[0]}+{len(labels) - 1}more"
    return tag


def _rel(path):
    return os.path.relpath(path, os.path.dirname(RESULT_ROOT))


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_dingo(path):
    """Posterior + weights + prior + t_ref + truth from a Dingo result."""
    result = Result(file_name=path)
    weights = (result.samples["weights"].to_numpy()
               if "weights" in result.samples else None)
    inference_parameters = result.metadata["train_settings"]["data"][
        "inference_parameters"]
    # t_ref, NOT event_metadata['time_event']: for a real event the two are the
    # same, but an injection has no time_event, and reading that key directly
    # left bilby's absolute GPS time unshifted -- a ~1.1e9 s error in t_c that
    # produced a silently wrong figure rather than an exception.
    return (result.samples, weights, result.prior, inference_parameters,
            result.t_ref)


def load_truth(path):
    """Injected truth as a dict, or None for a real event (which has none)."""
    with h5py.File(path, "r") as f:
        if "injection_parameters" not in f:
            return None
        return {k: float(f[f"injection_parameters/{k}"][()])
                for k in f["injection_parameters"]}


def load_bilby(path):
    """Posterior DataFrame from a bilby result (.json or .hdf5)."""
    if path.endswith(".json"):
        with open(path) as f:
            content = json.load(f)["posterior"]
        content = content.get("content", content)
        # bilby's posterior carries non-numeric columns too (waveform_approximant),
        # which cannot go into a float frame.
        numeric = {}
        for key, value in content.items():
            if np.ndim(value) != 1:
                continue
            try:
                numeric[key] = np.asarray(value, dtype=float)
            except (TypeError, ValueError):
                continue
        return pd.DataFrame(numeric)

    import bilby
    return bilby.core.result.read_in_result(filename=path).posterior


def align_time_convention(bilby_samples, t_ref):
    """Dingo stores geocent_time relative to t_ref; bilby stores absolute GPS."""
    if "geocent_time" not in bilby_samples or t_ref is None:
        return bilby_samples
    t = bilby_samples["geocent_time"].to_numpy()
    if np.abs(np.median(t)) < 1e5:
        return bilby_samples  # already relative
    bilby_samples = bilby_samples.copy()
    bilby_samples["geocent_time"] = t - float(t_ref)
    print(f"  bilby geocent_time converted from absolute GPS to an offset from "
          f"t_ref = {t_ref}")
    return bilby_samples


# --------------------------------------------------------------------------
# Discovery -- the naming convention, in one place
# --------------------------------------------------------------------------

def _model_label(path):
    """The model directory a posterior lives under: posteriors/<model>/<kind>/*.hdf5."""
    rel = os.path.relpath(path, POSTERIOR_DIR)
    if not rel.startswith(os.pardir):
        parts = rel.split(os.sep)
        # Compare mode uses posteriors/compare/<target>/<model>.hdf5 rather
        # than the usual posteriors/<model>/<kind>/<target>.hdf5 layout.
        if len(parts) >= 3 and parts[0] == "compare":
            return os.path.splitext(parts[-1])[0]
        return parts[0]
    return os.path.splitext(os.path.basename(path))[0]


def _target_of(path):
    """The target name a posterior file belongs to, inverting the conventions."""
    rel = os.path.relpath(path, POSTERIOR_DIR).split(os.sep)
    if len(rel) >= 3 and rel[0] == "compare":
        return rel[1]
    base = os.path.splitext(os.path.basename(path))[0]
    return base[: -len("_posterior")] if base.endswith("_posterior") else base


def _kind_of(path):
    """The posteriors/<model>/<kind>/ bucket a file sits in: real, single, series..."""
    rel = os.path.relpath(path, POSTERIOR_DIR).split(os.sep)
    return rel[1] if len(rel) > 2 else "other"


def available_targets():
    """{target: {model: path}} over every posterior under posteriors/."""
    found = {}
    for path in sorted(glob.glob(os.path.join(POSTERIOR_DIR, "*", "*", "*.hdf5"))):
        if path.endswith(".truth.json"):
            continue
        found.setdefault(_target_of(path), {})[_model_label(path)] = path
    return found


def _targets_by_kind():
    """{kind: {target: {model: path}}}, for a listing grouped the way the runs are."""
    grouped = {}
    for target, models in available_targets().items():
        kind = _kind_of(next(iter(models.values())))
        grouped.setdefault(kind, {})[target] = models
    return grouped


def find_bilby(target):
    """The bilby baseline for a target, or None. Real events and injections are
    stored under different roots, so try both rather than making the caller know."""
    candidates = [
        os.path.join(BILBY_EVENT_DIR, target, "bilby_result.json"),
        os.path.join(BILBY_INJECTION_DIR, target, "bilby_result.json"),
    ]
    candidates += sorted(glob.glob(
        os.path.join(BILBY_INJECTION_DIR, f"*{target}*", "bilby_result.json")))
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


# The parameter-sweep runs from injection_combo/ come in dozens at a time and are
# read with plot_marginals.py, not one corner at a time. Collapsing them keeps the
# listing to the handful of targets a corner is actually wanted for.
_BULK_KINDS = {"series"}

KIND_TITLES = {
    "real":   "REAL EVENTS      posteriors/<model>/real/",
    "single": "INJECTIONS       posteriors/<model>/single/",
    "series": "SWEEPS           posteriors/<model>/series/",
}


def print_listing(show_all=False):
    """What can be plotted right now, with the exact command to do it."""
    grouped = _targets_by_kind()
    if not grouped:
        raise SystemExit(f"No posteriors under {_rel(POSTERIOR_DIR)}.")

    print(f"\nPosteriors under {_rel(POSTERIOR_DIR)}")
    for kind in sorted(grouped, key=lambda k: (k in _BULK_KINDS, k)):
        targets = grouped[kind]
        print(f"\n{KIND_TITLES.get(kind, kind.upper())}")
        if kind in _BULK_KINDS and not show_all:
            names = sorted(targets)
            print(f"  {len(names)} sweep runs ({names[0]} ... {names[-1]}), "
                  f"--list --all to show them")
            print(f"  These are usually read with plot_marginals.py instead.")
            continue
        print(f"  {'target':<24}{'models':<44}{'bilby baseline'}")
        print(f"  {'-' * 88}")
        for target in sorted(targets):
            b = find_bilby(target)
            b_txt = _rel(b) if b else "-- none (Dingo-only figure)"
            print(f"  {target:<24}{', '.join(sorted(targets[target])):<44}{b_txt}")

    print(f"\nPlot one:         python plot_posteriors.py <target>")
    print(f"Choose models:    python plot_posteriors.py <target> --model <name> [<name> ...]")
    print(f"Choose params:    python plot_posteriors.py <target> -p chirp_mass chi_eff ...")
    print(f"Figures go to:    {_rel(FIGURES_DIR)}/<target>/\n")


def find_dingo_results(target, models=None, paths=None):
    """Resolve the Dingo posteriors to plot, as [(label, path), ...]."""
    if paths:
        return [(_model_label(p), p) for p in paths]

    targets = available_targets()
    if target not in targets:
        close = [t for t in targets if target.lower() in t.lower()]
        hint = f"\nDid you mean: {', '.join(sorted(close))}" if close else ""
        legacy = os.path.join(LEGACY_DIR, "real", f"{target}_result.hdf5")
        if os.path.exists(legacy):
            print(f"note: no current posterior for {target}; using the "
                  f"pre-reorganisation result {_rel(legacy)}")
            return [(_model_label(legacy), legacy)]
        raise SystemExit(
            f"No posterior for target {target!r}.{hint}\n"
            f"Run 'python plot_posteriors.py --list' to see what is available, or "
            f"produce it first from gnpe_model/inference/:\n"
            f"  MODE=real   EVENT={target} MODEL=<model> sbatch single_inference_submission.sh\n"
            f"  MODE=single SEED=<n>       MODEL=<model> sbatch single_inference_submission.sh"
        )

    by_model = targets[target]
    if models and list(models) != ["any"]:
        missing = [m for m in models if m not in by_model]
        if missing:
            raise SystemExit(
                f"{target}: no posterior for model(s) {', '.join(missing)}. "
                f"Available: {', '.join(sorted(by_model))}")
        return [(m, by_model[m]) for m in models]

    if len(by_model) == 1:
        return list(by_model.items())
    raise SystemExit(
        f"{len(by_model)} models have a posterior for {target}: "
        f"{', '.join(sorted(by_model))}.\n"
        f"Choose with --model <name> [<name> ...], or --model any to overlay all.")


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _resolve_parameters(requested, frames, where="every posterior"):
    """Keep the requested parameters that every posterior actually provides."""
    available = set(frames[0].columns)
    for frame in frames[1:]:
        available &= set(frame.columns)

    keep = [p for p in requested if p in available]
    dropped = [p for p in requested if p not in available]
    if dropped:
        print(f"  ! not sampled by {where}, dropping: {dropped}")
    if not keep:
        raise SystemExit(
            f"None of {requested} are available across {len(frames)} posterior(s); "
            f"the first has {sorted(frames[0].columns)}.")
    return keep


def _credible_contour_levels(density, fractions=CREDIBLE_LEVELS):
    flat = np.sort(density.ravel())[::-1]
    cumulative = np.cumsum(flat)
    cumulative /= cumulative[-1]
    levels = [flat[min(np.searchsorted(cumulative, f), len(flat) - 1)]
              for f in fractions]
    return sorted(set(levels))  # contour() needs strictly increasing levels


def _ranges(samples_list, parameters, truth=None, pad=0.05):
    """Axis ranges shared by every posterior, so panels are directly comparable."""
    ranges = {}
    for p in parameters:
        lo = min(np.percentile(s[p].to_numpy(), 0.5) for s in samples_list)
        hi = max(np.percentile(s[p].to_numpy(), 99.5) for s in samples_list)
        # A truth outside the posterior is the most important thing on the figure,
        # so widen rather than clip it off the edge.
        if truth is not None and p in truth:
            lo, hi = min(lo, truth[p]), max(hi, truth[p])
        span = hi - lo or 1.0
        ranges[p] = (lo - pad * span, hi + pad * span)
    return ranges


def _kde_1d(values, weights, grid):
    return gaussian_kde(values, weights=weights)(grid)


def _kde_2d(x, y, weights, gx, gy):
    kde = gaussian_kde(np.vstack([x, y]), weights=weights)
    X, Y = np.meshgrid(gx, gy)
    return X, Y, kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)


def _labels_for(prior):
    # Derived parameters first so a real prior entry always wins.
    return {**DERIVED_LATEX, **get_latex_labels(prior), "geocent_time": TIME_LABEL}


def kde_corner(samples_list, weights_list, labels, parameters, out,
               latex_labels=None, colors=COLORS, title=None, truth=None):
    """Corner drawn entirely from KDEs: smooth 1D curves on the diagonal, smooth
    credible-region contours off it. No histograms anywhere."""
    latex_labels = dict(latex_labels or {})
    latex_labels.setdefault("geocent_time", TIME_LABEL)
    n = len(parameters)
    ranges = _ranges(samples_list, parameters, truth=truth)
    grids = {p: np.linspace(*ranges[p], GRID) for p in parameters}

    fig, axes = plt.subplots(n, n, figsize=(2.7 * n, 2.7 * n))
    if n == 1:
        axes = np.array([[axes]])

    for i, pi in enumerate(parameters):
        for j, pj in enumerate(parameters):
            ax = axes[i][j]
            if j > i:
                ax.axis("off")
                continue

            peak = 0.0
            for k, (samples, weights) in enumerate(zip(samples_list, weights_list)):
                color = colors[k % len(colors)]
                if i == j:
                    density = _kde_1d(samples[pi].to_numpy(), weights, grids[pi])
                    ax.plot(grids[pi], density, color=color, lw=1.8)
                    ax.fill_between(grids[pi], density, color=color, alpha=0.12)
                    # Fix the y-range only once every curve is in, otherwise the
                    # first posterior's peak clips every later one.
                    peak = max(peak, float(density.max()))
                else:
                    X, Y, Z = _kde_2d(samples[pj].to_numpy(),
                                      samples[pi].to_numpy(), weights,
                                      grids[pj], grids[pi])
                    levels = _credible_contour_levels(Z)
                    # A posterior much narrower than the shared axis range puts
                    # nearly all of its KDE mass in one or two grid cells, so every
                    # credible level collapses onto Z.max() and the fill levels stop
                    # being strictly increasing. Drop the degenerate ones rather
                    # than letting contourf raise.
                    zmax = float(Z.max())
                    fill = [lev for lev in levels if lev < zmax] + [zmax]
                    if len(fill) >= 2:
                        ax.contourf(X, Y, Z, levels=fill,
                                    colors=[color], alpha=0.13)
                    line_levels = [lev for lev in levels if lev < zmax]
                    if line_levels:
                        ax.contour(X, Y, Z, levels=line_levels,
                                   colors=color, linewidths=1.4)
                    else:
                        # Unresolved on this grid: mark the peak so the posterior
                        # is not silently missing from the panel.
                        iy, ix = np.unravel_index(np.argmax(Z), Z.shape)
                        ax.plot(X[iy, ix], Y[iy, ix], marker="o", ms=3.5,
                                color=color, mew=0)
                    ax.set_ylim(*ranges[pi])

            if truth is not None:
                if pj in truth:
                    ax.axvline(truth[pj] if i != j else truth[pi],
                               color="k", lw=TRUTH_LW)
                if i != j and pi in truth:
                    ax.axhline(truth[pi], color="k", lw=TRUTH_LW)

            if i == j:
                ax.set_ylim(0, 1.08 * peak)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
                ax.set_ylabel("Probability density", fontsize=10)
            ax.set_xlim(*(ranges[pi] if i == j else ranges[pj]))

            if i == n - 1:
                ax.set_xlabel(latex_labels.get(pj, pj), fontsize=13)
                ax.tick_params(axis="x", labelrotation=45)
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(latex_labels.get(pi, pi), fontsize=13)

    handles = [Line2D([], [], color=colors[k % len(colors)], lw=2, label=lab)
               for k, lab in enumerate(labels)]
    if truth is not None:
        handles.append(Line2D([], [], color="k", lw=TRUTH_LW, label="injected truth"))
    fig.legend(handles=handles, loc="upper right",
               bbox_to_anchor=(0.98, 0.98), fontsize=15, frameon=False)
    if title:
        fig.suptitle(title, fontsize=16, x=0.5, y=0.995)
    contours = ", ".join(f"{int(100 * f)}%" for f in CREDIBLE_LEVELS)
    fig.text(0.98, 0.90, f"KDE; contours = {contours} credible",
             ha="right", fontsize=11, color="0.35")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dingo_only(dingo_runs, prior, parameters, out, target, truth=None):
    """Dingo posteriors on their own: one curve per model.

    A single model that carries importance-sampling weights is still shown twice
    (raw and reweighted), since that contrast is the point of the figure.
    """
    if len(dingo_runs) == 1:
        label, samples, weights = dingo_runs[0]
        samples = samples[parameters]
        if weights is not None:
            samples_list, weights_list, labels = (
                [samples, samples], [None, weights], ["Dingo", "Dingo-IS"])
        else:
            samples_list, weights_list, labels = [samples], [None], [f"Dingo {label}"]
    else:
        samples_list = [s[parameters] for _, s, _ in dingo_runs]
        weights_list = [w for _, _, w in dingo_runs]
        labels = [f"{lab} ({len(s)} samples)" for lab, s, _ in dingo_runs]

    kde_corner(samples_list, weights_list, labels, parameters, out,
               latex_labels=_labels_for(prior), title=f"{target} -- Dingo",
               truth=truth)
    print(f"Saved Dingo corner ({target}, {len(parameters)} params, "
          f"{len(samples_list)} posterior(s)) -> {_rel(out)}")


def plot_comparison(dingo_runs, prior, bilby_samples, parameters, out, target,
                    truth=None):
    """Every Dingo posterior overlaid on the bilby baseline."""
    samples_list = [s[parameters] for _, s, _ in dingo_runs]
    weights_list = [w for _, _, w in dingo_runs]
    labels = [f"Dingo {lab} ({len(s)} samples)" for lab, s, _ in dingo_runs]

    samples_list.append(bilby_samples[parameters])
    weights_list.append(None)
    labels.append(f"bilby ({len(bilby_samples)} samples)")

    kde_corner(samples_list, weights_list, labels, parameters, out,
               latex_labels=_labels_for(prior),
               title=f"{target} -- Dingo vs bilby", truth=truth)
    print(f"Saved comparison corner ({target}, {len(samples_list)} posteriors) "
          f"-> {_rel(out)}")


def plot_marginal_overlay(dingo_runs, prior, bilby_samples, parameters, out,
                          target, truth=None):
    """1D marginals of every Dingo posterior overlaid on the bilby baseline."""
    latex_labels = _labels_for(prior)
    n = len(parameters)

    curves = [(s, w, f"Dingo {lab}" if len(dingo_runs) > 1 else "Dingo")
              for lab, s, w in dingo_runs]
    curves.append((bilby_samples, None, "bilby"))
    ranges = _ranges([s[parameters] for s, _, _ in curves], parameters, truth=truth)

    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.2))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, parameters):
        grid = np.linspace(*ranges[p], GRID)
        for k, (samples, w, lab) in enumerate(curves):
            density = _kde_1d(samples[p].to_numpy(), w, grid)
            color = COLORS[k % len(COLORS)]
            ax.plot(grid, density, color=color, lw=2, label=lab)
            ax.fill_between(grid, density, color=color, alpha=0.15)
        if truth is not None and p in truth:
            ax.axvline(truth[p], color="k", lw=1.2, label="truth")
        ax.set_xlabel(latex_labels.get(p, p), fontsize=13)
        ax.set_yticks([])
        ax.set_ylim(bottom=0)
        ax.set_xlim(*ranges[p])
    axes[0].set_ylabel("posterior density", fontsize=12)
    axes[-1].legend(fontsize=12, frameon=False)
    fig.suptitle(f"{target} posterior estimation (bilby vs Dingo)", fontsize=15)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 1D marginal overlay ({target}) -> {_rel(out)}")


def print_summary(runs, parameters, truth=None, outpath=None):
    """Print and save posterior median and 90% CI."""

    header = f"{'parameter':<22}"
    if truth is not None:
        header += f"{'truth':>14}"
    for label, _ in runs:
        header += f"{label + ' median':>18}{label + ' 90% CI':>28}"

    rows = []

    for p in parameters:
        row = f"{p:<22}"
        if truth is not None:
            row += f"{truth.get(p, np.nan):>14.4f}"

        for _, samples in runs:
            lo, med, hi = np.percentile(samples[p], [5, 50, 95])
            row += f"{med:>18.4f}{f'[{lo:.4f}, {hi:.4f}]':>28}"

        rows.append(row)

    text = "\n".join([header, *rows])

    print("\n" + text)

    outpath = outpath or os.path.join(FIGURES_DIR, "summary.txt")
    with open(outpath, "w") as f:
        f.write(text + "\n")

    print(f"\nSummary table written to {_rel(outpath)}")


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("target", nargs="?", default=None,
                    help="Event or injection name, e.g. GW170809 or "
                         "seed0053_dprior. Run --list to see them. Omit only "
                         "when passing --dingo.")
    ap.add_argument("--list", action="store_true",
                    help="List every plottable target and exit.")
    ap.add_argument("--all", action="store_true",
                    help="With --list, also expand the sweep runs.")
    ap.add_argument("--model", nargs="+", default=None,
                    help="Which model's posterior(s) to plot. Several names are "
                         "overlaid; 'any' takes all. Default: the only model that "
                         "has this target, erroring if there is more than one.")
    ap.add_argument("--dingo", nargs="+", default=None,
                    help="Explicit Dingo hdf5 path(s), bypassing discovery.")
    ap.add_argument("--bilby", default=None,
                    help="Explicit bilby result (.json/.hdf5). Default: "
                         "auto-discovered. Pass 'none' for a Dingo-only figure.")
    ap.add_argument("-p", "--parameters", nargs="+", default=DEFAULT_PARAMETERS,
                    help=f"Default: {' '.join(DEFAULT_PARAMETERS)} "
                         f"(reduced to what every posterior provides).")
    ap.add_argument("--outdir", default=None,
                    help=f"Default: {_rel(FIGURES_DIR)}/. A subdirectory named "
                         f"after the target is created inside it.")
    args = ap.parse_args()

    if args.list:
        print_listing(show_all=args.all)
        return
    if not args.target and not args.dingo:
        ap.print_help()
        print("\nNothing to plot. Start with:  python plot_posteriors.py --list")
        raise SystemExit(1)

    found = find_dingo_results(args.target, args.model, args.dingo)
    target = args.target or _target_of(found[0][1])

    dingo_runs = []          # [(label, samples, weights), ...]
    prior, t_ref, truth = None, None, None
    for label, path in found:
        print(f"loading Dingo   [{label}] {_rel(path)}")
        samples, weights, prior_i, inference_parameters, t_ref_i = load_dingo(path)
        print(f"  model infers {len(inference_parameters)} parameters: "
              f"{inference_parameters}")
        if t_ref is not None and t_ref_i != t_ref:
            raise SystemExit(
                f"Posteriors disagree on the reference time ({t_ref} vs "
                f"{t_ref_i}); they are not the same event/injection.")
        # The prior only supplies LaTeX labels, so the first model's is enough.
        prior = prior if prior is not None else prior_i
        t_ref = t_ref_i
        truth = truth if truth is not None else load_truth(path)
        dingo_runs.append((label, samples, weights))
    if truth is not None:
        print(f"  injected truth found; drawn as a black cross")

    # Every figure of this run goes into a directory named after the target,
    # under figures/ or under --outdir, and carries the models it actually
    # contains. Without that tag every run of a given target writes the same
    # three filenames, so a two-model comparison silently replaced the
    # single-model figure -- and so did a second single-model run with a
    # different network.
    tag = _model_tag(dingo_runs)

    def out_path(name):
        name = f"{target}_{tag}_{name}"
        if args.outdir:
            directory = os.path.join(args.outdir, target)
            os.makedirs(directory, exist_ok=True)
            return os.path.join(directory, name)
        return _picture_path(name, subdir=target)

    dingo_frames = [s for _, s, _ in dingo_runs]
    parameters = _resolve_parameters(
        args.parameters, dingo_frames,
        where="every Dingo model" if len(dingo_runs) > 1 else "the Dingo model")
    plot_dingo_only(dingo_runs, prior, parameters,
                    out_path("dingo_corner.png"), target, truth=truth)

    if args.bilby is not None and args.bilby.lower() == "none":
        return
    bilby_path = args.bilby or find_bilby(target)
    if not bilby_path or not os.path.exists(bilby_path):
        print(f"\nNo bilby baseline for {target}; skipping the comparison figure.\n"
              f"Produce one from bilby_baseline/:\n"
              f"  RESULT={_rel(found[0][1])} sbatch submit_injection_baseline.sh\n"
              f"  EVENT={target} sbatch real_event.sh")
        return

    print(f"loading bilby   {_rel(bilby_path)}")
    bilby_samples = align_time_convention(load_bilby(bilby_path), t_ref)
    parameters = _resolve_parameters(args.parameters, dingo_frames + [bilby_samples])
    plot_comparison(dingo_runs, prior, bilby_samples, parameters,
                    out_path("dingo_vs_bilby.png"), target, truth=truth)
    plot_marginal_overlay(dingo_runs, prior, bilby_samples, parameters,
                          out_path("dingo_vs_bilby_1d.png"), target,
                          truth=truth)
    print_summary([(lab, s) for lab, s, _ in dingo_runs] + [("bilby", bilby_samples)],
                  parameters, truth=truth)


if __name__ == "__main__":
    main()
