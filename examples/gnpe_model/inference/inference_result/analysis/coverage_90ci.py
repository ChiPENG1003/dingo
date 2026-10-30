"""Paired marginal 90% equal-tail coverage, keyed by injection seed.

Example:
    python coverage_90ci.py --manifest pp_pairs100_20261023.csv --outdir ../tables/coverage_90ci
    from coverage_90ci import find_value
    event = find_value(10000)
    print(event['band_dingo'])

Manifest columns: seed,dingo,bilby. Each row selects one matched experiment.
Unpaired files and alternative model runs are deliberately not pooled together.
"""

import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from batch_read import load_injection_truth, load_injection_posterior
from plot_posteriors import load_bilby, align_time_convention
from dingo.gw.result import Result
from dingo.gw.derived_parameters import with_derived_parameters

DEFAULT_MANIFEST = Path(__file__).with_name('pp_pairs100_20261023.csv')
PARAMETERS = ['chirp_mass', 'mass_ratio', 'a_1', 'a_2', 'tilt_1', 'tilt_2',
              'phi_12', 'phi_jl', 'theta_jn', 'phase', 'luminosity_distance',
              'geocent_time', 'ra', 'dec', 'psi', 'chi_eff']


def read_manifest(manifest=DEFAULT_MANIFEST):
    path = Path(manifest).resolve()
    pairs = pd.read_csv(path)
    if not {'seed', 'dingo', 'bilby'}.issubset(pairs.columns):
        raise ValueError('Manifest must contain seed,dingo,bilby')
    if pairs.empty or pairs.seed.isna().any() or pairs.seed.duplicated().any():
        raise ValueError('Manifest is empty or has missing/duplicate seeds')
    if not np.equal(pairs.seed, pairs.seed.astype(int)).all():
        raise ValueError('Seeds must be integers')
    for column in ('dingo', 'bilby'):
        pairs[column] = pairs[column].map(lambda p: str((path.parent / p).resolve()))
    return pairs.sort_values('seed')


def interval_table(samples, truth):
    """Missing inference parameters are NA, never False; endpoints included."""
    if 'weights' in samples and not np.allclose(samples.weights, samples.weights.iloc[0]):
        raise ValueError('Nonuniform weights require weighted quantiles')
    rows = []
    for p in PARAMETERS:
        t = truth.get(p, np.nan)
        if p not in samples:
            rows.append((p, t, np.nan, np.nan, pd.NA, 0))
            continue
        values = samples[p].to_numpy(dtype=float)
        if not len(values) or not np.isfinite(values).all() or not np.isfinite(t):
            raise ValueError(f'Invalid posterior or truth for {p}')
        low, high = np.quantile(values, [0.05, 0.95])
        rows.append((p, t, low, high, bool(low <= t <= high), len(values)))
    table = pd.DataFrame(rows, columns=['parameter', 'truth', 'low', 'high', 'covered', 'n_samples']).set_index('parameter')
    table['covered'] = table['covered'].astype('boolean')
    return table


def _find_row(row):
    with h5py.File(row.dingo) as h:
        seed = int(h['injection_parameters'].attrs['injection_seed'])
        if seed != int(row.seed):
            raise ValueError(f'Seed mismatch: manifest {row.seed}, HDF5 {seed}')
    truth = with_derived_parameters(load_injection_truth(row.dingo), names=['chi_eff'])
    dingo = load_injection_posterior(row.dingo)
    bilby = align_time_convention(load_bilby(row.bilby), Result(file_name=row.dingo).t_ref)
    for samples in (dingo, bilby):
        if all(p in samples for p in ('mass_ratio','a_1','a_2','tilt_1','tilt_2')):
            samples['chi_eff'] = with_derived_parameters(samples, names=['chi_eff'])['chi_eff']
    d, b = interval_table(dingo, truth), interval_table(bilby, truth)
    valid = d.covered.notna() & b.covered.notna()
    only_d = (d.covered & ~b.covered).where(valid, pd.NA)
    only_b = (b.covered & ~d.covered).where(valid, pd.NA)
    return dict(seed=seed, truth_value=truth, dingo_result=d, bilby_result=b,
                band_dingo=d.covered, band_bilby=b.covered,
                band_only_dingo=only_d, band_only_bilby=only_b)


def find_value(seed, manifest=DEFAULT_MANIFEST):
    pairs = read_manifest(manifest)
    selected = pairs[pairs.seed == seed]
    if len(selected) != 1:
        raise KeyError(f'Seed {seed} not found in selected manifest')
    return _find_row(next(selected.itertuples(index=False)))


def summarize(manifest=DEFAULT_MANIFEST):
    pairs = read_manifest(manifest)
    details = []
    for row in pairs.itertuples(index=False):
        event = _find_row(row)
        for p in PARAMETERS:
            d, b = event['dingo_result'].loc[p], event['bilby_result'].loc[p]
            details.append(dict(seed=int(row.seed), parameter=p, truth=d.truth,
                dingo_low=d.low, dingo_high=d.high, bilby_low=b.low, bilby_high=b.high,
                dingo_covered=d.covered, bilby_covered=b.covered,
                only_dingo=event['band_only_dingo'][p], only_bilby=event['band_only_bilby'][p]))
        print('Processed seed', row.seed, flush=True)
    detail = pd.DataFrame(details)
    metrics = []
    for p in PARAMETERS:
        group = detail[detail.parameter == p]
        nd, nb = group.dingo_covered.notna().sum(), group.bilby_covered.notna().sum()
        valid = group.dingo_covered.notna() & group.bilby_covered.notna()
        n = int(valid.sum())
        dc, bc = int(group.dingo_covered.dropna().sum()), int(group.bilby_covered.dropna().sum())
        od, ob = int(group.only_dingo.dropna().sum()), int(group.only_bilby.dropna().sum())
        if n == nd == nb:
            assert dc-bc == od-ob
        metrics.append(dict(parameter=p,n_injections=len(pairs),n_dingo=int(nd),n_bilby=int(nb),n_paired=n,
            dingo_count=dc if nd else np.nan,bilby_count=bc if nb else np.nan,
            only_dingo_count=od if n else np.nan,only_bilby_count=ob if n else np.nan,
            dingo_coverage=dc/nd if nd else np.nan,bilby_coverage=bc/nb if nb else np.nan,
            only_dingo_fraction=od/n if n else np.nan,only_bilby_fraction=ob/n if n else np.nan))
    return pd.DataFrame(metrics), detail


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest',default=str(DEFAULT_MANIFEST))
    ap.add_argument('--outdir',required=True)
    args = ap.parse_args()
    summary, detail = summarize(args.manifest)
    out = Path(args.outdir); out.mkdir(parents=True,exist_ok=True)
    summary.to_csv(out/'coverage_summary.csv',index=False)
    detail.to_csv(out/'coverage_per_seed.csv',index=False)
    (out/'method.txt').write_text('90% CI = posterior quantiles [0.05,0.95], including endpoints.\n'
        'Coverage is calculated per parameter across seeds, not across parameters within a seed.\n'
        'Missing parameters are NA. Counts and actual denominators are saved explicitly.\n'
        'Only-method fractions use the paired available seeds. Current manifest has 100 pairs.\n'
        'chi_eff is derived; geocent_time is relative to Dingo t_ref. Angles use raw coordinate equal-tail intervals.\n'
        'Current 100-injection selection excludes failed seed10026; this is a selected-sample analysis.\n')
    print(summary.to_string(index=False),flush=True)

if __name__ == '__main__':
    main()
