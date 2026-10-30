"""Audit one cell end to end: recompute everything from the raw stored samples.

Every number on a heat map passes through four stages, and each one can be
checked independently of the code that produced it:

  1. the distraction signal was built as the axis specified   -> theta2 vs the bin
  2. the SNR it was supposed to have is the SNR it has        -> snr2 vs snr2_target
  3. the truth quantile stored per run really is the quantile -> recomputed from samples
  4. the cell's colour really is n_covered / n_total          -> recounted here

Stage 3 is the one worth caring about: it re-reads the posterior samples and
recomputes the quantile from scratch, so a bug anywhere in _truth_quantiles or
the storage path shows up as a mismatch.

  python verify_cell.py population --cell 0 0
  python verify_cell.py snr_ratio --cell 0 0 --level 0.9 --recompute-snr
"""

import argparse
import os

import h5py
import numpy as np

import combo_common as cc


def _theta1_from(cell_file):
    return {k: float(v) for k, v in cell_file["theta1"].attrs.items()}


def verify(sweep, x_index, y_index, level=0.9, parameter="chirp_mass",
           recompute_snr=False, show=3):
    config = cc.read_json(os.path.join(cc.sweep_dir(sweep), "config.json"))
    path = cc.cell_path(sweep, x_index, y_index)
    if not os.path.exists(path):
        raise SystemExit(f"No such cell: {cc.rel(path)}")

    lo, hi = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    print(f"cell {cc.rel(path)}")
    print(f"sweep axis      {config['axis']}")
    print(f"shift (y)       {config['time_shifts'][y_index]:.4f} s   "
          f"(signal 2 merges this much EARLIER)")
    if config["axis_centers"]:
        print(f"x bin centre    {config['axis_centers'][x_index]:.6g}")

    with h5py.File(path, "r") as f:
        theta1 = _theta1_from(f)
        runs = sorted(k for k in f if k.startswith("run_"))
        print(f"runs in cell    {len(runs)}")
        print(f"SNR_1           {f.attrs['snr1']:.4f}  (fixed for the whole study)")
        print(f"theta1          Mc {theta1['chirp_mass']:.4f}  q "
              f"{theta1['mass_ratio']:.4f}  d_L {theta1['luminosity_distance']:.1f}  "
              f"t_c {theta1['geocent_time']:.6f}")

        recomputed, stored, covered = [], [], 0
        snr_err = []
        theta2_rows = []

        for name in runs:
            g = f[name]
            columns = list(g["samples"].attrs["columns"])
            samples = np.asarray(g["samples"])
            if parameter not in columns:
                raise SystemExit(f"{parameter!r} not stored; have {columns}")
            column = samples[:, columns.index(parameter)].astype(float)

            # Stage 3: the quantile, from the samples, not from the attribute.
            q_recomputed = float(np.mean(column < float(theta1[parameter])))
            recomputed.append(q_recomputed)
            stored.append(float(g.attrs[f"quantile_{parameter}"]))
            if lo < q_recomputed < hi:
                covered += 1

            if "snr2_target" in g.attrs:
                snr_err.append(float(g.attrs["snr2"]) - float(g.attrs["snr2_target"]))

            if len(theta2_rows) < show and "theta2_chirp_mass" in g.attrs:
                theta2_rows.append((
                    name,
                    float(g.attrs["theta2_chirp_mass"]),
                    float(g.attrs["theta2_mass_ratio"]),
                    float(g.attrs["theta2_luminosity_distance"]),
                    float(g.attrs["theta2_geocent_time"]),
                    float(g.attrs.get("snr2", float("nan"))),
                ))

    recomputed = np.array(recomputed)
    stored = np.array(stored)

    print(f"\nfirst {len(theta2_rows)} distraction signals")
    print(f"  {'run':10s} {'Mc2':>8s} {'q2':>7s} {'d_L2':>9s} {'t_c2':>10s} {'SNR_2':>8s}")
    for row in theta2_rows:
        print(f"  {row[0]:10s} {row[1]:8.3f} {row[2]:7.3f} {row[3]:9.1f} "
              f"{row[4]:10.6f} {row[5]:8.3f}")

    # Stage 1: geocent_time of signal 2 must equal signal 1's, or the y axis is
    # smeared by the +-0.1 s prior on it.
    ties = [abs(r[4] - theta1["geocent_time"]) for r in theta2_rows]
    print(f"\n[1] t_c tie to signal 1     max |diff| = {max(ties):.3e}  "
          f"{'OK' if max(ties) < 1e-9 else 'BROKEN'}")

    # Stage 2: SNR solve landed where the axis asked.
    if snr_err:
        snr_err = np.abs(np.array(snr_err))
        print(f"[2] SNR_2 vs its target     max |diff| = {snr_err.max():.3e}  "
              f"{'OK' if snr_err.max() < 1e-6 else 'OFF'}")
    else:
        print("[2] SNR_2 not pinned on this axis (population / distance sweep) "
              "-- nothing to check")

    # Stage 3: stored quantiles reproduce.
    delta = np.abs(recomputed - stored).max()
    print(f"[3] quantile recomputed     max |diff| = {delta:.3e}  "
          f"{'OK' if delta < 1e-9 else 'MISMATCH'}")

    # Stage 4: the cell value.
    print(f"[4] coverage recounted      {covered}/{len(recomputed)} = "
          f"{covered / len(recomputed):.4f}   at the {level:.0%} level")
    print(f"    quantile spread         min {recomputed.min():.3f}  "
          f"median {np.median(recomputed):.3f}  max {recomputed.max():.3f}")

    if recompute_snr:
        print("\n[extra] regenerating waveforms to re-measure SNR_2 from scratch ...")
        label, main_path, _ = cc.resolve_model(config["model"].split("=")[0]
                                               if "=" in config["model"]
                                               else config["model"])
        ig = cc.injection_generator_from_metadata(cc.metadata_only(main_path))
        shift = config["time_shifts"][y_index]
        with h5py.File(path, "r") as f:
            for name in runs[:show]:
                g = f[name]
                theta2 = {k[len("theta2_"):]: float(v) for k, v in g.attrs.items()
                          if k.startswith("theta2_")}
                snr2, _ = cc.snr_of(ig, theta2, shift)
                print(f"  {name}  stored {float(g.attrs['snr2']):.4f}  "
                      f"regenerated {snr2:.4f}  diff {abs(snr2 - float(g.attrs['snr2'])):.2e}")

    return covered / len(recomputed)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep")
    p.add_argument("--cell", type=int, nargs=2, metavar=("X", "Y"), default=(0, 0))
    p.add_argument("--level", type=float, default=0.9)
    p.add_argument("--parameter", default="chirp_mass")
    p.add_argument("--recompute-snr", action="store_true",
                   help="Regenerate the waveforms and re-measure SNR_2. Slow "
                        "(a few seconds per run) but checks the SNR path too.")
    p.add_argument("--show", type=int, default=3)
    args = p.parse_args(argv)
    return verify(args.sweep, args.cell[0], args.cell[1], level=args.level,
                  parameter=args.parameter, recompute_snr=args.recompute_snr,
                  show=args.show)


if __name__ == "__main__":
    main()
