#!/usr/bin/env python
import os
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bilby
from dingo.gw.result import Result

# Parameters compared (present in both posteriors). geocent_time is handled
# separately because of the relative/absolute convention.
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


def js_divergence_nat(x, y, bins=40, weights_x=None, weights_y=None):
    from scipy.spatial.distance import jensenshannon

    lo = min(np.min(x), np.min(y))
    hi = max(np.max(x), np.max(y))
    edges = np.linspace(lo, hi, bins + 1)

    p, _ = np.histogram(x, bins=edges, weights=weights_x)
    q, _ = np.histogram(y, bins=edges, weights=weights_y)

    p = p / p.sum()
    q = q / q.sum()

    d = jensenshannon(p, q, base=np.e)
    return float(d ** 2)


def load_dingo(path):
    """Return (DataFrame with relative geocent_time, weights or None, t_ref)."""
    r = Result(file_name=path)
    # r.samples is already a DataFrame; np.array() on it would drop the column
    # names and leave an integer-indexed frame that matches no parameter.
    df = pd.DataFrame(r.samples).copy()
    weights = df["weights"].to_numpy() if "weights" in df.columns else None
    return df, weights, r.t_ref


def load_bilby(path, t_ref):
    """Return (DataFrame with relative geocent_time, weights or None)."""
    res = bilby.core.result.read_in_result(filename=path)
    df = res.posterior.copy()
    if "geocent_time" in df.columns:
        df["geocent_time"] = df["geocent_time"] - t_ref      # -> relative
    return df, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dingo", required=True, help="Dingo result.hdf5")
    ap.add_argument("--bilby", required=True, help="bilby *_result.hdf5/json")
    ap.add_argument("--outdir", default="comparison")
    ap.add_argument("--bins", type=int, default=100)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    dingo_df, dingo_w, t_ref = load_dingo(args.dingo)
    bilby_df, bilby_w = load_bilby(args.bilby, t_ref)

    params = [p for p in COMMON_PARAMS
              if p in dingo_df.columns and p in bilby_df.columns]
    if "geocent_time" in dingo_df.columns and "geocent_time" in bilby_df.columns:
        params.append("geocent_time")

    # --- JS divergence table ---
    print(f"\n{'parameter':>20} {'JS (bits)':>12}")
    js = {}
    for p in params:
        js[p] = js_divergence_nat(
            dingo_df[p].to_numpy(), bilby_df[p].to_numpy(),
            bins=args.bins, weights_x=dingo_w, weights_y=bilby_w,
        )
        print(f"{p:>20} {js[p]:>12.4f}")
    print(f"\n{'median JS':>20} {np.median(list(js.values())):>12.4f}")
    print(f"{'max JS':>20} {max(js.values()):>12.4f}  ({max(js, key=js.get)})")
    with open(os.path.join(args.outdir, "js_divergence.txt"), "w") as f:
        for p in params:
            f.write(f"{p}\t{js[p]:.6f}\n")

    # --- overlaid corner plot ---
    import corner
    labels = [LATEX.get(p, p) for p in params]
    dingo_arr = dingo_df[params].to_numpy()
    bilby_arr = bilby_df[params].to_numpy()
    rng = [
        (min(dingo_arr[:, i].min(), bilby_arr[:, i].min()),
         max(dingo_arr[:, i].max(), bilby_arr[:, i].max()))
        for i in range(len(params))
    ]
    fig = corner.corner(
        dingo_arr, labels=labels, color="C0", range=rng,
        weights=dingo_w, plot_datapoints=False, plot_density=False,
        levels=(0.5, 0.9), hist_kwargs=dict(density=True),
    )
    corner.corner(
        bilby_arr, fig=fig, color="C1", range=rng,
        weights=bilby_w, plot_datapoints=False, plot_density=False,
        levels=(0.5, 0.9), hist_kwargs=dict(density=True),
    )
    # legend
    fig.axes[0].plot([], [], color="C0", label="Dingo (NPE)")
    fig.axes[0].plot([], [], color="C1", label="bilby (baseline)")
    fig.legend(loc="upper right", fontsize=14)
    out = os.path.join(args.outdir, "dingo_vs_bilby_corner.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"\nSaved corner plot -> {out}")
    print(f"Saved JS table   -> {os.path.join(args.outdir, 'js_divergence.txt')}")


if __name__ == "__main__":
    main()
