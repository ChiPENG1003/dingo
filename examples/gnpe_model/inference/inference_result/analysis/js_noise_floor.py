#!/usr/bin/env python
"""Finite-sample noise floor for the JS divergences reported by compare.py.

A histogram JS divergence between two *finite* sample sets is biased upward:
even when both sets come from the same distribution, empty and unevenly filled
bins make JS > 0. With only a few hundred samples per posterior that bias can
be comparable to the signal, so a raw JS value is not interpretable on its own.

This script measures the bias empirically. For each parameter it draws two
independent bootstrap resamples from the *same* parent (the Dingo samples), at
the same sizes as the real comparison, and computes their JS. The median of
those trials is the value one would expect if Dingo and bilby agreed perfectly;
the 95th percentile is a rough significance threshold. Whatever the observed JS
exceeds that by is real disagreement.

Sweeping --bins is the other half of the check: a real difference is stable
against bin count, a binning artefact is not.

Units are nats, matching compare.py.

Usage mirrors compare.py:

    python js_noise_floor.py seed0053_dprior --model ET_best
"""
import argparse
import sys

import numpy as np

from compare import COMMON_PARAMS, common_parameters, js_divergence_nat
from plot_posteriors import (
    _rel,
    _target_of,
    align_time_convention,
    find_bilby,
    find_dingo_results,
    load_bilby,
    load_dingo,
)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", nargs="?", default=None,
                    help="Event or injection name, e.g. seed0053_dprior.")
    ap.add_argument("--model", nargs="+", default=None,
                    help="Which model's posterior to check. One at a time.")
    ap.add_argument("--dingo", nargs="+", default=None,
                    help="Explicit Dingo hdf5 path, bypassing discovery.")
    ap.add_argument("--bilby", default=None,
                    help="Explicit bilby result. Default: auto-discovered.")
    ap.add_argument("--bins", type=int, nargs="+", default=[100, 40, 15],
                    help="Bin counts to sweep. A real difference is stable "
                         "across these; a binning artefact is not.")
    ap.add_argument("--trials", type=int, default=300,
                    help="Bootstrap trials per parameter per bin count.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not args.target and not args.dingo:
        ap.print_help()
        raise SystemExit(1)

    found = find_dingo_results(args.target, args.model, args.dingo)
    if len(found) > 1:
        raise SystemExit(
            f"The noise floor is measured for one posterior at a time; got "
            f"{len(found)}: {', '.join(label for label, _ in found)}.")
    label, path = found[0]
    target = args.target or _target_of(path)

    print(f"loading Dingo   [{label}] {_rel(path)}")
    dingo_df, dingo_w, _, _, t_ref = load_dingo(path)

    bilby_path = args.bilby or find_bilby(target)
    if not bilby_path:
        raise SystemExit(f"No bilby baseline for {target}.")
    print(f"loading bilby   {_rel(bilby_path)}")
    bilby_df = align_time_convention(load_bilby(bilby_path), t_ref)

    params = common_parameters([dingo_df, bilby_df])
    if not params:
        sys.exit("No parameters in common between the two posteriors.")

    n_d, n_b = len(dingo_df), len(bilby_df)
    rng = np.random.default_rng(args.seed)
    print(f"\nn_dingo={n_d}  n_bilby={n_b}  trials={args.trials}  "
          f"(JS in nats)\n")

    for bins in args.bins:
        print(f"--- bins = {bins} ---")
        print(f"{'parameter':>20} {'JS obs':>9} {'null med':>9} {'null 95%':>9} "
              f"{'excess':>9}")
        for p in params:
            x = dingo_df[p].to_numpy()
            y = bilby_df[p].to_numpy()
            obs = js_divergence_nat(x, y, bins=bins)
            # Null hypothesis: both sets drawn from the same distribution, at
            # the same sizes as the real comparison. The Dingo samples serve as
            # the common parent (the choice of parent barely matters -- the
            # bias is driven by sample size and bin count, not by the shape).
            null = np.array([
                js_divergence_nat(rng.choice(x, n_d, replace=True),
                                  rng.choice(x, n_b, replace=True), bins=bins)
                for _ in range(args.trials)])
            med, hi = np.median(null), np.percentile(null, 95)
            print(f"{p:>20} {obs:>9.4f} {med:>9.4f} {hi:>9.4f} {obs - med:>9.4f}")
        print()


if __name__ == "__main__":
    main()
