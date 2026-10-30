"""Choose signal 1, and check that the grid it implies is actually reachable.

Run this BEFORE any sweep. It needs no GPU: the checkpoint is opened with a lazy
mmap for its metadata only, and everything here is waveform generation and
algebra on the CPU.

Three things get settled:

  1. which seed gives a target signal with a sane mass ratio and SNR -- q_1 caps
     the chirp mass axis (q_1 = 0.5 would cap it at 48.7 rather than 65);
  2. the sign convention of the time shift, measured rather than assumed;
  3. whether the SNR ratio axis needs distances outside the trained prior
     U(100, 2000) Mpc.

  python select_target.py --model 1M_7params_O2 --seeds 0 1 2 3 4 --target-snr1 25
"""

import argparse

import numpy as np

import combo_common as cc
import grid_axes as ga
import inference as inf


def _candidate_row(ig, seed, target_snr1):
    theta = inf._draw_reproducible_truth(ig, seed)
    snr_prior, _ = cc.snr_of(ig, theta)
    d_prior = float(theta["luminosity_distance"])
    q1 = float(theta["mass_ratio"])
    mc_range = cc.chirp_mass_range_for_q(q1)
    d_needed = d_prior * snr_prior / target_snr1 if target_snr1 else d_prior
    return {
        "seed": seed,
        "chirp_mass": float(theta["chirp_mass"]),
        "mass_ratio": q1,
        "distance_prior": d_prior,
        "snr_at_prior_distance": snr_prior,
        "distance_for_target_snr": d_needed,
        "chirp_mass_axis_max": None if mc_range is None else mc_range[1],
        "theta": theta,
    }


def build_signal1(ig, seed, target_snr1=None):
    """Signal 1, optionally moved to an exact network SNR.

    Pinning SNR_1 is what makes the SNR ratio axis a round number rather than
    whatever the prior draw happened to give, and it fixes the absolute scale of
    every heat map in the study.
    """
    theta = inf._draw_reproducible_truth(ig, seed)
    if target_snr1:
        distance, _ = cc.distance_for_snr(ig, theta, target_snr1)
        theta["luminosity_distance"] = float(distance)
        if not 100.0 <= distance <= 2000.0:
            print(f"WARNING: signal 1 needs d_L = {distance:.1f} Mpc to reach "
                  f"SNR {target_snr1:g}, outside the trained prior U(100, 2000) "
                  f"Mpc. The network has not seen a target at this distance, so "
                  f"its coverage will be off before any distraction signal is added.")
    snr1, snr1_det = cc.snr_of(ig, theta)
    return theta, snr1, snr1_det


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=inf._env("MODEL", str, inf.DEFAULT_MODEL))
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    p.add_argument("--target-snr1", type=float, default=25.0,
                   help="Move signal 1 to this network SNR. 0 keeps the prior draw.")
    p.add_argument("--choose", type=int, default=None,
                   help="Report the full resolved grid for this seed.")
    p.add_argument("--num-bins", type=int, default=10)
    args = p.parse_args(argv)

    label, main_path, init_path = cc.resolve_model(args.model)
    metadata = cc.metadata_only(main_path)
    ig = cc.injection_generator_from_metadata(metadata)

    print(f"model         {label}")
    print(f"checkpoint    {cc.rel(main_path)}")
    print(f"infers        {', '.join(cc.inference_parameters(metadata))}")
    print(f"detectors     {', '.join(sorted(ig.asd))}")
    print(f"domain        f in [{ig.data_domain.f_min:g}, {ig.data_domain.f_max:g}] Hz, "
          f"delta_f = {ig.data_domain.delta_f:g} Hz, T = {1/ig.data_domain.delta_f:g} s")

    # ---- 1. shift direction, measured -------------------------------------
    probe = inf._draw_reproducible_truth(ig, args.seeds[0])
    moved = cc.verify_shift_direction(ig, probe, shift=0.5)
    direction = "LATER" if moved > 0 else "EARLIER"
    print(f"\nshift check   _linear_time_shift(+0.5 s) moves the merger "
          f"{abs(moved):.4f} s {direction}")
    print(f"              -> the y axis is |dt| with signal 2 merging {direction.lower()} "
          f"than signal 1")

    # ---- 2. candidate target signals --------------------------------------
    print(f"\ncandidate signal 1 (target SNR {args.target_snr1:g}):")
    print(f"  {'seed':>5} {'Mc':>8} {'q':>7} {'d_prior':>9} {'SNR@prior':>10} "
          f"{'d needed':>9} {'Mc axis max':>12}")
    rows = []
    for seed in args.seeds:
        r = _candidate_row(ig, seed, args.target_snr1)
        rows.append(r)
        cap = "none" if r["chirp_mass_axis_max"] is None else f"{r['chirp_mass_axis_max']:.1f}"
        flag = "" if 100 <= r["distance_for_target_snr"] <= 2000 else "  <- outside prior"
        print(f"  {seed:5d} {r['chirp_mass']:8.2f} {r['mass_ratio']:7.3f} "
              f"{r['distance_prior']:9.1f} {r['snr_at_prior_distance']:10.2f} "
              f"{r['distance_for_target_snr']:9.1f} {cap:>12}{flag}")
    print("\n  'Mc axis max' is the largest chirp mass compatible with THAT seed's q,")
    print("  and therefore the ceiling of the chirp mass axis (requested: 65).")

    if args.choose is None:
        best = max(rows, key=lambda r: (r["chirp_mass_axis_max"] or 0))
        print(f"\n  widest chirp mass axis: seed {best['seed']} "
              f"(q = {best['mass_ratio']:.3f}, cap {best['chirp_mass_axis_max']:.1f})")
        print(f"  rerun with --choose {best['seed']} to see the full grid.")
        return rows

    # ---- 3. the resolved grid for one chosen seed -------------------------
    seed = args.choose
    theta1, snr1, snr1_det = build_signal1(ig, seed, args.target_snr1)
    print(f"\n{'=' * 74}\nchosen signal 1: seed {seed}")
    inf._print_truth(theta1, "")
    print(f"  network SNR = {snr1:.3f}   " +
          "  ".join(f"{d} {v:.2f}" for d, v in sorted(snr1_det.items())))

    ctx = ga.Context(ig, theta1, snr1, snr1_det)
    print(f"\ntime shift axis (y, shared): " +
          ", ".join(f"{t:.3f}" for t in cc.TIME_SHIFTS) + "  s")

    for name in ga.AXIS_NAMES:
        try:
            axis = ga.build_axis(name, ctx, num_bins=args.num_bins)
        except ValueError as e:
            print(f"\naxis {name}: UNAVAILABLE -- {e}")
            continue
        print(f"\naxis {name}")
        print("  centres " + ", ".join(f"{c:.3g}" for c in axis.centers))
        for note in axis.notes:
            print(f"  note    {note}")
        if axis.target_snr2 is not None:
            print(f"  SNR_2   pinned to {axis.target_snr2:.2f} "
                  f"(ratio {snr1 / axis.target_snr2:.2f})")

        # What distance does each bin actually need? Probed on one realisation
        # at the shortest and longest separation, which bracket the truncation
        # effect of the non-circular shift.
        needed = []
        for x_index in range(len(axis.centers)):
            for shift in (cc.TIME_SHIFTS[0], cc.TIME_SHIFTS[-1]):
                theta2, info = ga.build_theta2(ctx, axis, x_index, shift, realization=0)
                needed.append(float(theta2["luminosity_distance"]))
        lo, hi = min(needed), max(needed)
        outside = [d for d in needed if not 100 <= d <= 2000]
        print(f"  d_2     {lo:.1f} - {hi:.1f} Mpc" +
              (f"   !! {len(outside)}/{len(needed)} probes outside the trained "
               f"prior U(100, 2000)" if outside else ""))

    print(f"\n{'=' * 74}")
    print("If that all looks right, launch a sweep with e.g.\n"
          f"  sbatch submit_grid.sh snr_ratio\n"
          f"(set TARGET_SEED={seed} TARGET_SNR1={args.target_snr1:g} in the submit script)")
    return rows


if __name__ == "__main__":
    main()
