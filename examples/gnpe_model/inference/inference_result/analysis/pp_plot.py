import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kstest, combine_pvalues

import coverage_stats as cs
from batch_read import (
    INJECTIONS_DIR,
    list_injection_files,
    load_injection_truth,
    load_injection_posterior,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.dirname(HERE)
FIGURES_DIR = os.path.join(RESULT_ROOT, "figures")
TABLES_DIR = os.path.join(RESULT_ROOT, "tables")
Path(TABLES_DIR).mkdir(exist_ok=True)
Path(FIGURES_DIR).mkdir(exist_ok=True)


def credible_levels(injections_dir):
    """{parameter: array of one credible level per injection}."""
    files = list_injection_files(injections_dir)
    rows = []
    for i, path in enumerate(files):
        index = int(os.path.splitext(os.path.basename(path))[0].split("_")[-1])
        truth = load_injection_truth(path)
        posterior = load_injection_posterior(path)
        for p in posterior.columns:
            if p not in truth:
                continue
            post = posterior[p].to_numpy()
            post = post[np.isfinite(post)]
            if len(post) == 0:
                continue
            rows.append({"index": index, "parameter": p,
                         "credible_level": float(np.mean(post < truth[p]))})
        if (i + 1) % 100 == 0 or (i + 1) == len(files):
            print(f"  {i + 1}/{len(files)} injections", flush=True)
    return pd.DataFrame(rows)


def plot_pp(df, filename, title, confidence=(0.68, 0.95, 0.997)):
    params = list(df["parameter"].drop_duplicates())
    n = df["index"].nunique()
    x = np.linspace(0, 1, 1001)

    fig, ax = plt.subplots(figsize=(7, 7))
    # Binomial confidence bands: at each x the count below x is Binomial(n, x),
    # so the band is the pointwise quantile of that distribution.
    from scipy.stats import binom
    for level, alpha in zip(confidence, (0.16, 0.10, 0.05)):
        lo = binom.ppf(0.5 - level / 2, n, x) / n
        hi = binom.ppf(0.5 + level / 2, n, x) / n
        ax.fill_between(x, lo, hi, color="grey", alpha=alpha, linewidth=0)

    pvalues = []
    for p in params:
        v = np.sort(df.loc[df["parameter"] == p, "credible_level"].to_numpy())
        pv = kstest(v, "uniform").pvalue
        pvalues.append(pv)
        ax.plot(np.concatenate([[0], v, [1]]),
                np.concatenate([[0], np.arange(1, len(v) + 1) / len(v), [1]]),
                linewidth=1.4, label=f"{p}  ({pv:.3f})")

    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("credible level")
    ax.set_ylabel("fraction of injections with truth inside")
    # Fisher's method over parameters is only indicative: the parameters share
    # the same injections, so their KS p-values are not independent.
    combined = combine_pvalues(pvalues)[1]
    ax.set_title(f"{title}\n{n} injections (combined p = {combined:.4f}) ")
    ax.legend(fontsize=8, loc="lower right", title="parameter (KS p)")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")
    return dict(zip(params, pvalues)), combined


def compare_paired(manifest, outdir):
    """Compare paired Dingo/Bilby posterior CDF ranks; retain legacy CLI below."""
    import h5py
    from scipy.stats import binom
    from dingo.gw.result import Result
    from dingo.gw.derived_parameters import with_derived_parameters
    from plot_posteriors import load_bilby, align_time_convention
    pairs = pd.read_csv(manifest)
    if pairs.seed.duplicated().any():
        raise ValueError('Duplicate injection seeds in manifest')
    params = ['chirp_mass', 'mass_ratio', 'chi_eff', 'luminosity_distance',
              'a_1', 'a_2', 'tilt_1', 'tilt_2', 'geocent_time', 'ra', 'dec', 'psi']
    rows = []
    for pair in pairs.itertuples():
        with h5py.File(pair.dingo) as h:
            assert int(h['injection_parameters'].attrs['injection_seed']) == pair.seed
        truth = with_derived_parameters(load_injection_truth(pair.dingo), names=['chi_eff'])
        dingo = load_injection_posterior(pair.dingo)
        dingo['chi_eff'] = with_derived_parameters(dingo, names=['chi_eff'])['chi_eff']
        result = Result(file_name=pair.dingo)
        bilby = align_time_convention(load_bilby(pair.bilby), result.t_ref)
        bilby['chi_eff'] = with_derived_parameters(bilby, names=['chi_eff'])['chi_eff']
        for method, posterior in [('Dingo', dingo), ('Bilby', bilby)]:
            if 'weights' in posterior and not np.allclose(posterior.weights, posterior.weights.iloc[0]):
                raise ValueError('Non-uniform weights require weighted ranks')
            for p in params:
                values = posterior[p].to_numpy(dtype=float)
                if not len(values) or not np.isfinite(values).all() or not np.isfinite(truth[p]):
                    raise ValueError(f'Invalid samples/truth: {pair.seed}, {method}, {p}')
                rows.append(dict(seed=pair.seed, method=method, parameter=p,
                                 credible_level=float(np.mean(values < truth[p])), n_samples=len(values)))
        print('Paired seed', pair.seed, flush=True)
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out/'credible_levels_paired.csv', index=False)
    pairs.to_csv(out/'injection_pairs.csv', index=False)
    n = len(pairs)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharex=True, sharey=True)
    x = np.linspace(0, 1, 1001)
    stats = []
    for ax, p in zip(axes.flat, params):
        for level, alpha in zip((0.997, 0.95, 0.68), (0.07, 0.12, 0.18)):
            ax.fill_between(x, binom.ppf((1-level)/2,n,x)/n,
                            binom.ppf((1+level)/2,n,x)/n, color='grey', alpha=alpha, lw=0)
        ax.plot([0,1],[0,1], color='black', linestyle=':', lw=1)
        for method, color, style in [('Dingo','#1b7f79','-'), ('Bilby','#d1622b','--')]:
            v = np.sort(df.loc[(df.parameter==p)&(df.method==method),'credible_level'])
            assert len(v)==n
            ks = kstest(v,'uniform')
            stats.append(dict(method=method,parameter=p,n=n,ks_statistic=ks.statistic,pvalue=ks.pvalue))
            ax.step(np.r_[0,v,1],np.r_[0,np.arange(1,n+1)/n,1],where='post',
                    color=color,ls=style,lw=1.6,label=f'{method} (KS p={ks.pvalue:.3g})')
        ax.set_title(p)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.legend(fontsize=8,loc='upper left')
        ax.grid(alpha=0.15)
    fig.supxlabel('Posterior CDF rank threshold')
    fig.supylabel('Fraction of injections with rank below threshold')
    fig.suptitle(f'Dingo 5M complete best model vs Bilby (nlive=250) | {n} paired injections\n'
                 'Pointwise binomial 68/95/99.7% bands; seed 10026 excluded after inference failure',fontsize=13)
    fig.tight_layout(rect=(0.02,0.02,1,0.94))
    for ext in ('png','pdf'):
        fig.savefig(out/f'pp_dingo_vs_bilby_100.{ext}',dpi=180)
    plt.close(fig)
    pd.DataFrame(stats).to_csv(out/'ks_summary.csv',index=False)
    (out/'README.txt').write_text(
        'Posterior CDF rank = mean(posterior sample < injected truth).\n'
        'Identical paired injections and parameters for both methods.\n'
        'chi_eff derived from joint component-spin samples in both methods.\n'
        'Bilby absolute GPS times shifted to Dingo reference time.\n'
        'Seeds 10000 through 10100 except 10026; exclusion after sampling failure introduces selection.\n'
        'This selected sample is not an unfiltered prior-predictive calibration experiment.\n'
        'Bands are pointwise binomial, not simultaneous. KS p-values are nominal: finite posterior samples,\n'
        'multiple parameters and selection should be considered; no combined p-value is reported.\n'
        'Angular parameters use the existing coordinate ordering, not HPD coverage.\n')
    print('COMPARISON COMPLETE',out,flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--injections-dir", default=INJECTIONS_DIR)
    ap.add_argument("--label", default="")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument('--paired-manifest', help='CSV with seed,dingo,bilby columns')
    ap.add_argument('--comparison-outdir', default=os.path.join(FIGURES_DIR, 'pp_comparison'))
    args = ap.parse_args()
    if args.paired_manifest:
        compare_paired(args.paired_manifest, args.comparison_outdir)
        return

    tag = f"_{args.label}" if args.label else ""
    cache = os.path.join(TABLES_DIR, f"credible_levels{tag}.csv")
    if os.path.exists(cache) and not args.rebuild:
        print(f"Loading cached credible levels from {cache}")
        df = pd.read_csv(cache)
    else:
        print(f"Reading injections from {args.injections_dir} ...")
        df = credible_levels(args.injections_dir)
        df.to_csv(cache, index=False)
        print(f"Saved {cache} ({len(df)} rows)")

    title = "Dingo P-P plot" + (f" [{args.label}]" if args.label else "")
    pvalues, combined = plot_pp(df, os.path.join(FIGURES_DIR, f"pp_plot{tag}.png"), title)

    lines = [f"=== {title} ({df['index'].nunique()} injections) ===",
             "KS test of the credible levels against U(0,1); small p = miscalibrated",
             ""]
    for p, pv in pvalues.items():
        lines.append(f"  {p:22s} p = {pv:.4f}")
    lines.append(f"\nCombined (Fisher, indicative only): p = {combined:.4f}")
    # psi and ra are periodic; the credible level uses the raw ordering, so a
    # posterior that wraps around the boundary is scored as if it did not.
    lines.append("\nNote: ra and psi are periodic and the credible level is "
                 "computed on raw values, so a posterior straddling the "
                 "boundary is penalised spuriously.")
    text = "\n".join(lines)
    out = os.path.join(TABLES_DIR, f"pp_summary{tag}.txt")
    with open(out, "w") as f:
        f.write(text + "\n")
    print("\n" + text)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
