
import os
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

CREDIBLE_LEVEL = 0.90
LOWER_PERCENTILE = 100 * (1 - CREDIBLE_LEVEL) / 2
UPPER_PERCENTILE = 100 * (1 + CREDIBLE_LEVEL) / 2

# Long-form cache columns (no spaces, stable across Dingo/bilby).
RECORD_COLUMNS = ["index", "parameter", "peak", "median", "truth",
                  "lower", "upper", "radius90", "residual", "ratio", "in90",
                  "truth_source"]

# Ratio-histogram x-range (kept identical for both datasets so bars line up).
# Values beyond it are explicitly accumulated into edge bins and labelled.
RATIO_RANGE = (-1.0, 5.0)

# Parameters whose coordinates wrap around after one period.  psi is defined
# modulo pi; the other phase/sky-longitude variables are modulo 2 pi.
CIRCULAR_PERIODS = {
    "phi_12": 2 * np.pi,
    "phi_jl": 2 * np.pi,
    "ra": 2 * np.pi,
    "psi": np.pi,
}


# --------------------------------------------------------------------------- #
# Per-injection reduction
# --------------------------------------------------------------------------- #
def find_peak(post: np.ndarray, parameter: str, n_grid: int = 2000) -> float:
    post = np.asarray(post)
    post = post[np.isfinite(post)]
    if len(post) == 0:
        return np.nan
    if np.std(post) == 0:
        return float(post[0])
    kde = gaussian_kde(post)
    period = CIRCULAR_PERIODS.get(parameter)
    if period is not None:
        # Unwrap around the circular mean before estimating the mode.  This
        # turns samples just below 2 pi and just above 0 into one local mode.
        phase = 2 * np.pi * post / period
        centre = np.angle(np.mean(np.exp(1j * phase))) * period / (2 * np.pi)
        centre %= period
        unwrapped = centre + parameter_difference(post, centre, parameter)
        kde = gaussian_kde(unwrapped)
        x = np.linspace(centre - period / 2, centre + period / 2, n_grid)
        density = kde(x)
        return float(x[np.argmax(density)] % period)
    else:
        x = np.linspace(post.min(), post.max(), n_grid)
        density = kde(x)
    return float(x[np.argmax(density)])


def parameter_difference(value, reference, parameter: str):
    """Signed physical difference, with shortest-path differences for angles."""
    difference = np.asarray(value) - reference
    period = CIRCULAR_PERIODS.get(parameter)
    if period is not None:
        difference = (difference + period / 2) % period - period / 2
    return difference


def subsample(posterior: pd.DataFrame, n_samples: int | None,
              seed: int = 0) -> pd.DataFrame:
    """Randomly subsample a posterior to n_samples rows (for matching Dingo and
    bilby to a common per-injection sample count). None or n>=len -> unchanged."""
    if n_samples is None or n_samples >= len(posterior):
        return posterior
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(posterior), size=n_samples, replace=False)
    return posterior.iloc[idx]


def analyze_posterior(posterior: pd.DataFrame, truth: dict,
                      n_samples: int | None = None) -> dict:
    """Reduce one posterior to per-parameter rows (only params present in truth).

    n_samples: if set, randomly subsample the posterior to this many samples
    first, so Dingo and bilby are compared with the same per-injection count.
    """
    posterior = subsample(posterior, n_samples)
    rows = {}
    for p in posterior.columns:
        if p not in truth:
            continue
        post = posterior[p].to_numpy()
        post = post[np.isfinite(post)]
        if len(post) == 0:
            continue
        lower, upper = np.percentile(post, [LOWER_PERCENTILE, UPPER_PERCENTILE])
        med = float(np.median(post))
        tv = truth[p]
        peak = find_peak(post, p)
        # This region contains 90% of posterior samples in the physical metric
        # centred on the mode.  It remains meaningful for signed, near-zero and
        # periodic parameters, unlike the relative error (median - truth) / truth.
        sample_distance = np.abs(parameter_difference(post, peak, p))
        radius90 = float(np.percentile(sample_distance, 100 * CREDIBLE_LEVEL))
        delta = float(parameter_difference(peak, tv, p))
        rows[p] = {
            "peak": peak,
            "median": med,
            "truth": tv,
            "lower": float(lower),
            "upper": float(upper),
            "radius90": radius90,
            "residual": delta / radius90 if radius90 > 0 else np.nan,
            # Relative error of the posterior MEDIAN, not the KDE mode: the median
            # is invariant under monotonic reparametrisation, needs no bandwidth,
            # and is what the P-P plot's credible level of 0.5 refers to.
            "ratio": _relative_error(med, tv),
            "in90": bool(abs(delta) <= radius90),
        }
    return rows


def _relative_error(estimate, truth):
    """(estimate - truth) / truth; NaN where the truth is zero."""
    estimate = np.asarray(estimate, dtype=float)
    truth = np.asarray(truth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (estimate - truth) / truth
    return np.where(truth != 0, out, np.nan) if out.ndim else (
        float(out) if truth != 0 else np.nan)


def rows_to_records(index: int, rows: dict,
                    truth_source: str = "embedded_hdf5") -> list[dict]:
    """Flatten analyze_posterior() output into long-form records for caching."""
    return [
        {"index": int(index), "parameter": p, **row,
         "truth_source": truth_source}
        for p, row in rows.items()
    ]


# --------------------------------------------------------------------------- #
# Cache I/O (long-form CSV: one row per injection x parameter)
# --------------------------------------------------------------------------- #
def save_records(records: list[dict], cache_path: str) -> pd.DataFrame:
    df = pd.DataFrame(records)[RECORD_COLUMNS]
    df.to_csv(cache_path, index=False)
    print(f"Saved analysis cache -> {cache_path}  ({len(df)} rows)")
    return df


def load_records(cache_path: str) -> pd.DataFrame:
    df = pd.read_csv(cache_path)
    missing = set(RECORD_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Analysis cache {cache_path} uses an obsolete schema "
            f"(missing {sorted(missing)})."
        )
    df["in90"] = df["in90"].astype(bool)
    # Caches written before the ratio switched from the KDE peak to the median
    # still carry the peak-based value in this column. Every input needed is
    # stored, so re-derive it rather than trust the column or force a rebuild.
    df["ratio"] = _relative_error(df["median"].to_numpy(), df["truth"].to_numpy())
    return df


def ratios_from_df(df: pd.DataFrame) -> dict:
    """{parameter: [(median - truth) / truth, ...]} in first-appearance order.

    Meaningful only for positive, scale-free parameters (chirp_mass, mass_ratio,
    luminosity_distance); for signed or periodic ones the denominator is small or
    arbitrary.
    """
    out = {}
    for p in df["parameter"].drop_duplicates():
        out[p] = df.loc[df["parameter"] == p, "ratio"].to_numpy()
    return out


def coverage_from_df(df: pd.DataFrame) -> dict:
    out = {}
    for p in df["parameter"].drop_duplicates():
        out[p] = df.loc[df["parameter"] == p, "in90"].to_numpy()
    return out


def n_injections_from_df(df: pd.DataFrame) -> int:
    return int(df["index"].nunique())


# --------------------------------------------------------------------------- #
# Coverage summary (the print_coverage_summary content, now saveable)
# --------------------------------------------------------------------------- #
def summarize_coverage(coverage: dict, n_injections: int) -> pd.DataFrame:
    rows = {}
    for p, flags in coverage.items():
        flags = np.asarray(flags, dtype=bool)
        rows[p] = {
            "in 90% CI": int(flags.sum()),
            "total": int(flags.size),
            "coverage": flags.sum() / flags.size if flags.size else np.nan,
        }
    return pd.DataFrame(rows).T[["in 90% CI", "total", "coverage"]]


def format_coverage_summary(summary: pd.DataFrame, n_injections: int,
                            title: str = "90% mode-centred region coverage") -> str:
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    lines = [
        f"=== {title} ({n_injections} injections) ===",
        "(a well-calibrated posterior should give coverage close to 0.90)",
        "",
        summary.to_string(),
    ]
    inside = summary["in 90% CI"].sum()
    total = summary["total"].sum()
    lines.append(f"\nOverall: {int(inside)}/{int(total)} = {inside / total:.4f}")
    return "\n".join(lines)


def save_coverage_summary(summary: pd.DataFrame, n_injections: int,
                          txt_path: str, csv_path: str | None = None,
                          title: str = "90% mode-centred region coverage") -> str:
    text = format_coverage_summary(summary, n_injections, title=title)
    with open(txt_path, "w") as f:
        f.write(text + "\n")
    if csv_path:
        summary.to_csv(csv_path)
    print(text)
    print(f"\nSaved coverage summary -> {txt_path}"
          + (f" and {csv_path}" if csv_path else ""))
    return text


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _histogram_values(values: np.ndarray, plot_range: tuple[float, float]):
    """Keep all observations visible by accumulating overflows at the edges."""
    lo, hi = plot_range
    below = int(np.sum(values < lo))
    above = int(np.sum(values > hi))
    # Stay infinitesimally inside the final bins, so values at the boundary are
    # not lost to Matplotlib's half-open bin convention.
    display = np.clip(values, np.nextafter(lo, hi), np.nextafter(hi, lo))
    return display, below, above


def plot_ratio_histograms(ratios: dict, filename: str,
                          bins: int = 40, ncols: int = 4) -> None:
    """Histograms of the relative error of the posterior median."""
    params = list(ratios.keys())
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    flat = axes.flatten()
    for ax, p in zip(flat, params):
        v = np.asarray(ratios[p], dtype=float)
        v = v[np.isfinite(v)]
        display, below, above = _histogram_values(v, RATIO_RANGE)
        ax.hist(display, bins=bins, range=RATIO_RANGE, color="steelblue",
                edgecolor="white")
        ax.axvline(0.0, color="crimson", linestyle="--", linewidth=1)
        ax.set_xlim(*RATIO_RANGE)
        ax.set_title(p)
        ax.set_xlabel("ratio (median - truth) / truth")
        ax.set_ylabel("counts")
        if below or above:
            ax.text(0.98, 0.96, f"≤{RATIO_RANGE[0]:g}: {below}  "
                    f"≥{RATIO_RANGE[1]:g}: {above}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7)
    for ax in flat[len(params):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")


def plot_ratio_overlay(ratios_a: dict, ratios_b: dict,
                       label_a: str, label_b: str, filename: str,
                       bins: int = 40, ncols: int = 4) -> None:
    """Overlay two datasets' relative-error histograms.

    Only parameters present in BOTH are plotted. Densities (not counts) are
    used so different injection counts are still comparable.
    """
    params = [p for p in ratios_a if p in ratios_b]
    nrows = math.ceil(len(params) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False)
    flat = axes.flatten()
    for ax, p in zip(flat, params):
        va = np.asarray(ratios_a[p], dtype=float)
        vb = np.asarray(ratios_b[p], dtype=float)
        va = va[np.isfinite(va)]
        vb = vb[np.isfinite(vb)]
        da, below_a, above_a = _histogram_values(va, RATIO_RANGE)
        db, below_b, above_b = _histogram_values(vb, RATIO_RANGE)
        ax.hist(da, bins=bins, range=RATIO_RANGE, density=True, histtype="step",
                color="C0", linewidth=1.8, label=f"{label_a} (n={len(va)})")
        ax.hist(db, bins=bins, range=RATIO_RANGE, density=True, histtype="step",
                color="C1", linewidth=1.8, label=f"{label_b} (n={len(vb)})")
        ax.axvline(0.0, color="crimson", linestyle="--", linewidth=1)
        ax.set_xlim(*RATIO_RANGE)
        ax.set_title(p)
        ax.set_xlabel("ratio (median - truth) / truth")
        ax.set_ylabel("density")
        if below_a or above_a or below_b or above_b:
            ax.text(0.98, 0.96,
                    f"{label_a}: {below_a}/{above_a}; {label_b}: {below_b}/{above_b}",
                    transform=ax.transAxes, ha="right", va="top", fontsize=6)
        ax.legend(fontsize=7)
    for ax in flat[len(params):]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)
    print(f"Saved {filename}")
