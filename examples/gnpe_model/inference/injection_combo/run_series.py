"""Ten hand-picked overlapping injections, each with its own full posterior.

The heat maps in this package answer "how often is the truth covered" over 100
noise realisations per cell. This script answers the other question: what does
ONE posterior look like as the distraction signal is varied? Ten injections, one
result file each, all sharing the same signal 1 and the same noise:

  01  reference          signal 1 alone, no distraction signal
  02-04  |dt| = 0.5, 0.2, 0.1 s     signal 2 exactly as the prior drew it
  05-07  |dt| = 0.2 s, Mc_2 = 20, 50, 100 Msun
  08-10  |dt| = 0.2 s, d_L,2 = 100, 1000, 2000 Mpc

Signal 1 is the study's target (select_target.build_signal1: seed 1 moved to
network SNR 25), so these ten sit on the same footing as every heat map cell.
Signal 2 is realisation 0's nuisance draw with its geocent_time tied to signal
1's, so |dt| IS the merger separation, and then one parameter pinned per file.

The noise seed is identical in all ten, so 02-10 differ from 01 by exactly the
distraction signal and from each other by exactly the pinned parameter.

  python run_series.py --dry-run     # parameters, SNRs, output names; no GPU
  python run_series.py               # the ten inferences (GPU)
  python run_series.py --only 05 07  # redo two of them
"""

import argparse
import json
import os
import time
import types

import numpy as np
import torch

import combo_common as cc
import inference as inf
from select_target import build_signal1

# Realisation 0 of the grids: same nuisance draw for signal 2, same noise.
NUISANCE_SEED = 200000
NOISE_SEED = 300000

TARGET_SEED = 1
TARGET_SNR1 = 25.0
DEFAULT_MODEL = "1M_7params_O2"

# Network SNR of the untouched signal-2 draw at |dt| = 0.2 s. The separation
# entries below are pinned to it so that they are directly comparable with the
# time entries, which use that draw unchanged.
SERIES_SNR2 = 10.44

# name -> (shift, overrides). The shift is the separation in seconds
# (_linear_time_shift moves signal 2 EARLIER for a positive value; see
# combo_common.verify_shift_direction). None means "no second signal".
SERIES = [
    ("01_reference", None, {}),
    ("02_dt0.5s", 0.5, {}),
    ("03_dt0.2s", 0.2, {}),
    ("04_dt0.1s", 0.1, {}),
    ("05_dt0.2s_mc20", 0.2, {"second_chirp_mass": 20.0}),
    ("06_dt0.2s_mc50", 0.2, {"second_chirp_mass": 50.0}),
    ("07_dt0.2s_mc100", 0.2, {"second_chirp_mass": 100.0}),
    ("08_dt0.2s_dl100", 0.2, {"second_distance": 100.0}),
    ("09_dt0.2s_dl1000", 0.2, {"second_distance": 1000.0}),
    ("10_dt0.2s_dl2000", 0.2, {"second_distance": 2000.0}),
    # Added after the first ten: the tightest separation the grids use (their y
    # axis starts at 0.05 s). Numbered at the end rather than slotted into the
    # time group, so the ten files already on disk keep their names.
    ("11_dt0.05s", 0.05, {}),
    # Tighter than the grids go, and tighter than the 10 ms light travel time
    # between the detectors: the two mergers now sit inside one another's
    # response. The shift is quantised to the sample rate (0.01 s -> 20 samples
    # at 2048 Hz = 0.00977 s); verify_series.py measures what was realised.
    ("12_dt0.01s", 0.01, {}),
    # Source-separation axis. The 3-D separation is NOT something the data
    # responds to -- h1 + h2 responds to signal 2's amplitude, arrival time and
    # sky position -- so it cannot be varied on its own: it is a function of the
    # radial distance and of the sky angle. The distance sweep above already
    # covers the radial half, with SNR_2 locked to it (61 -> 3). These entries
    # cover the angular half with SNR_2 held FIXED, so whatever they show is
    # geometry rather than loudness. gamma = 92.7 deg is where the prior drew
    # signal 2, i.e. entry 03; 15_ lands back on it as a consistency check.
    ("13_sep_gamma0", 0.2, {"sky_separation_deg": 0.0, "target_snr2": SERIES_SNR2}),
    ("14_sep_gamma30", 0.2, {"sky_separation_deg": 30.0, "target_snr2": SERIES_SNR2}),
    ("15_sep_gamma90", 0.2, {"sky_separation_deg": 90.0, "target_snr2": SERIES_SNR2}),
    ("16_sep_gamma180", 0.2, {"sky_separation_deg": 180.0, "target_snr2": SERIES_SNR2}),
    # SNR ratio axis, the grids' main one: SNR_1 is fixed at 25, so pinning
    # SNR_2 = 25 / ratio through the 1/d solve makes the ratio exact per
    # realisation. Five log-spaced points across [0.1, 10]; ratio < 1 means the
    # intruder is the LOUDER of the two. The distances this needs at the loud end
    # are far inside the trained prior's 100 Mpc floor -- fine for a signal
    # nobody models, but it is why d_L,2 is reported next to the ratio.
    *[(f"{17 + i}_ratio{ratio:g}", 0.2,
       {"target_snr2": TARGET_SNR1 / ratio, "axis": "snr_ratio"})
      for i, ratio in enumerate(np.geomspace(0.1, 10.0, 5))],
    # Two more points on the same axis, appended rather than renumbered: 0.1 and
    # 1 are already 17_ and 19_.
    *[(f"{22 + i}_ratio{ratio:g}", 0.2,
       {"target_snr2": TARGET_SNR1 / ratio, "axis": "snr_ratio"})
      for i, ratio in enumerate((0.5, 5.0, 0.3, 0.4))],
    # ra and dec of the intruder, one at a time, with its distance left alone.
    # The gamma entries above rotate the sky position AND re-solve the distance
    # to hold SNR_2 fixed; these do not, so they carry the antenna response with
    # them -- at fixed distance SNR_2 swings by a factor ~3.7 over the sky.
    *[(f"{26 + i}_ra{ra:.3g}", 0.2, {"second_ra": ra, "axis": "ra"})
      for i, ra in enumerate(np.arange(4) * np.pi / 2)],
    *[(f"{30 + i}_dec{dec:+.2g}", 0.2, {"second_dec": dec, "axis": "dec"})
      for i, dec in enumerate((-1.2, -0.4, 0.4, 1.2))],
    # Filling both coordinates out to their full domain: the four points above
    # left the polar caps (|dec| > 1.2, 7% of the sky) unsampled and put only
    # four points around the ra period. ra: the odd multiples of pi/4, giving
    # eight in all; dec: the ends and the equator, giving -1.5 ... +1.5.
    *[(f"{34 + i}_ra{ra:.3g}", 0.2, {"second_ra": ra, "axis": "ra"})
      for i, ra in enumerate((np.arange(4) * 2 + 1) * np.pi / 4)],
    *[(f"{38 + i}_dec{dec:+.2g}", 0.2, {"second_dec": dec, "axis": "dec"})
      for i, dec in enumerate((-1.5, -0.75, 0.0, 0.75, 1.5))],
]


# Which sweep an entry belongs to, keyed by its leading number. Entries added
# later say so in their overrides instead ({"axis": ...}); this table covers the
# ones written before the field existed.
AXIS_OF = {**{f"{i:02d}": "shift" for i in (2, 3, 4, 11, 12)},
           **{f"{i:02d}": "chirp_mass" for i in (5, 6, 7)},
           **{f"{i:02d}": "luminosity_distance" for i in (8, 9, 10)},
           **{f"{i:02d}": "separation" for i in (13, 14, 15, 16)}}


def axis_of(name):
    return AXIS_OF.get(name[:2])


def series_dir(model_label):
    return inf.mode_dir("series", model_label)


def _place_at_sky_separation(theta2, theta1, gamma_deg):
    """Move signal 2 to angular separation `gamma_deg` from signal 1.

    It travels along the great circle it already sits on -- the bearing from
    signal 1 to signal 2's drawn position is kept -- so gamma = gamma_0 puts it
    back exactly where the prior drew it, and the axis is a pure rotation
    towards or away from the target rather than a jump to an unrelated patch.
    """
    ra1, dec1 = float(theta1["ra"]), float(theta1["dec"])
    ra2, dec2 = float(theta2["ra"]), float(theta2["dec"])
    delta_ra = ra2 - ra1
    bearing = np.arctan2(
        np.sin(delta_ra) * np.cos(dec2),
        np.cos(dec1) * np.sin(dec2) - np.sin(dec1) * np.cos(dec2) * np.cos(delta_ra),
    )
    gamma = np.radians(float(gamma_deg))
    new_dec = np.arcsin(np.sin(dec1) * np.cos(gamma)
                        + np.cos(dec1) * np.sin(gamma) * np.cos(bearing))
    new_ra = ra1 + np.arctan2(
        np.sin(bearing) * np.sin(gamma) * np.cos(dec1),
        np.cos(gamma) - np.sin(dec1) * np.sin(new_dec),
    )
    theta2["ra"] = float(new_ra % (2 * np.pi))
    theta2["dec"] = float(new_dec)


def build_theta2(ig, theta1, overrides, shift_seconds=None):
    """Signal 2 for one entry: realisation 0's draw, tied in time, then pinned.

    Parameter pinning goes through inference._apply_second_overrides so that this
    script and `inference.py overlap --second-chirp-mass ...` cannot drift apart
    on what "pin a parameter" means. Two controls live here instead, because they
    are not single-parameter edits: the sky rotation, and the distance solve that
    holds SNR_2 fixed while the sky position changes the antenna response.
    """
    overrides = dict(overrides)
    overrides.pop("axis", None)          # metadata, not a parameter to set
    gamma_deg = overrides.pop("sky_separation_deg", None)
    second_ra = overrides.pop("second_ra", None)
    second_dec = overrides.pop("second_dec", None)
    target_snr2 = overrides.pop("target_snr2", None)

    theta2 = inf._draw_reproducible_truth(ig, NUISANCE_SEED)
    pinned = {"free_second_time": False, "second_chirp_mass": 0.0,
              "second_distance": 0.0, **overrides}
    theta2, notes = inf._apply_second_overrides(
        theta2, theta1, types.SimpleNamespace(**pinned))

    if gamma_deg is not None:
        _place_at_sky_separation(theta2, theta1, gamma_deg)
        notes.append(f"sky separation from signal 1 set to {gamma_deg:g} deg")
    # ra and dec set directly, one at a time: unlike the gamma rotation these
    # leave the distance alone, so the antenna response -- and with it SNR_2 --
    # moves with the sky position. That is the effect being measured.
    if second_ra is not None:
        theta2["ra"] = float(second_ra) % (2 * np.pi)
        notes.append(f"ra set to {theta2['ra']:.4f} rad")
    if second_dec is not None:
        theta2["dec"] = float(second_dec)
        notes.append(f"dec set to {theta2['dec']:.4f} rad")
    if target_snr2:
        # After the shift, since the non-circular shift truncates inspiral and
        # so changes the SNR that will actually be realised.
        distance, _ = cc.distance_for_snr(ig, theta2, target_snr2,
                                          shift_seconds=shift_seconds)
        theta2["luminosity_distance"] = float(distance)
        notes.append(f"luminosity_distance solved to {distance:.1f} Mpc for "
                     f"SNR_2 = {target_snr2:g}")
    return theta2, notes


def describe(ig, theta1, snr1):
    """One row per entry: what is injected and how loud it is."""
    delta_f = ig.data_domain.delta_f
    rows = []
    for name, shift, overrides in SERIES:
        row = {"name": name, "shift_seconds": shift, "pinned": dict(overrides),
               # Which quantity this entry is a point of. Recorded so a plot does
               # not have to guess the axis from which numbers happen to differ.
               "axis": overrides.get("axis") or axis_of(name)}
        if shift is None:
            row.update(theta2=None, snr2=None, snr_ratio=None)
        else:
            theta2, notes = build_theta2(ig, theta1, overrides, shift_seconds=shift)
            waveform2, asds = cc.projected_signal(ig, theta2, shift)
            snr2, snr2_det = cc.network_snr(waveform2, asds, delta_f)
            m1, m2 = cc.component_masses(theta2["chirp_mass"], theta2["mass_ratio"])
            row.update(theta2=theta2, notes=notes, snr2=snr2,
                       snr2_detectors=snr2_det, snr_ratio=snr1 / snr2,
                       mass_1_2=float(m1), mass_2_2=float(m2),
                       # Derived, and recorded on every entry rather than only on
                       # the separation ones: any file can then be placed on the
                       # separation axis afterwards.
                       sky_separation_deg=float(cc.sky_separation(
                           theta1["ra"], theta1["dec"], theta2["ra"], theta2["dec"])),
                       source_separation_mpc=float(cc.source_separation(
                           theta1["luminosity_distance"], theta1["ra"], theta1["dec"],
                           theta2["luminosity_distance"], theta2["ra"], theta2["dec"])))
        rows.append(row)
    return rows


def print_table(rows, theta1, snr1, snr1_det, out_dir, target_seed, target_snr1):
    m1, m2 = cc.component_masses(theta1["chirp_mass"], theta1["mass_ratio"])
    print(f"\nsignal 1 (seed {target_seed}, moved to network SNR {target_snr1:g}):")
    for key in ("chirp_mass", "mass_ratio", "luminosity_distance", "geocent_time",
                "theta_jn", "ra", "dec", "psi"):
        print(f"   {key:22s} {theta1[key]: .6f}")
    print(f"   m1 = {m1:.2f}, m2 = {m2:.2f} Msun")
    print(f"   network SNR = {snr1:.2f}  "
          + "  ".join(f"{d}={v:.2f}" for d, v in sorted(snr1_det.items())))

    print(f"\n{'file':22s} {'|dt| s':>7s} {'Mc_2':>7s} {'q_2':>6s} {'m1_2':>7s} "
          f"{'m2_2':>7s} {'dL_2':>7s} {'SNR_2':>7s} {'SNR1/SNR2':>10s}")
    print("-" * 96)
    for row in rows:
        if row["theta2"] is None:
            print(f"{row['name']:22s} {'-':>7s}   (no second signal)")
            continue
        t2 = row["theta2"]
        print(f"{row['name']:22s} {row['shift_seconds']:7.2f} {t2['chirp_mass']:7.2f} "
              f"{t2['mass_ratio']:6.3f} {row['mass_1_2']:7.2f} {row['mass_2_2']:7.2f} "
              f"{t2['luminosity_distance']:7.1f} {row['snr2']:7.2f} "
              f"{row['snr_ratio']:10.2f}")
    print(f"\noutputs -> {cc.rel(out_dir)}/<file>.hdf5 (+ .truth.json)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=inf._env("MODEL", str, DEFAULT_MODEL))
    p.add_argument("--target-seed", type=int, default=TARGET_SEED)
    p.add_argument("--target-snr1", type=float, default=TARGET_SNR1,
                   help="Network SNR signal 1 is moved to. 0 keeps the prior draw.")
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-iterations", type=int, default=30)
    p.add_argument("--dry-run", action="store_true",
                   help="Parameters, SNRs and output names only. No GPU needed.")
    p.add_argument("--only", nargs="+", default=None, metavar="PREFIX",
                   help="Run just these entries, by name or leading number.")
    p.add_argument("--force", action="store_true",
                   help="Redo entries whose output file already exists.")
    args = p.parse_args(argv)

    label, main_path, init_path = cc.resolve_model(args.model)
    out_dir = series_dir(label)

    if args.dry_run:
        # Metadata-only: builds the prior and the waveforms without CUDA.
        metadata = cc.metadata_only(main_path)
        ig = cc.injection_generator_from_metadata(metadata)
    else:
        main_pm, init_pm = inf._load_pair(main_path, init_path)
        ig = inf.build_injection_generator(main_pm)
        metadata = main_pm.metadata

    theta1, snr1, snr1_det = build_signal1(ig, args.target_seed, args.target_snr1)
    rows = describe(ig, theta1, snr1)
    print_table(rows, theta1, snr1, snr1_det, out_dir,
                args.target_seed, args.target_snr1)

    manifest = {
        "model": label,
        "model_main": cc.rel(main_path),
        "target_seed": args.target_seed,
        "target_snr1": args.target_snr1,
        "theta1": theta1,
        "snr1": snr1,
        "snr1_detectors": snr1_det,
        "nuisance_seed": NUISANCE_SEED,
        "noise_seed": NOISE_SEED,
        "shift_convention": "positive |dt| = signal 2 merges EARLIER than signal 1",
        "num_samples": args.num_samples,
        "entries": [{k: v for k, v in row.items() if k != "notes"} for row in rows],
    }
    if args.dry_run:
        print("\n--dry-run: nothing written, no inference run.")
        return manifest

    os.makedirs(out_dir, exist_ok=True)
    cc.write_json(os.path.join(out_dir, "series.json"), manifest)

    selected = [r for r in rows
                if args.only is None
                or any(str(o) == r["name"] or r["name"].startswith(str(o))
                       for o in args.only)]
    if not selected:
        raise SystemExit(f"--only {args.only} matched nothing. "
                         f"Names: {[r['name'] for r in rows]}")

    sampler = inf.build_sampler(main_pm, init_pm, num_iterations=args.num_iterations)
    batch_size = args.batch_size or args.num_samples
    parameters = cc.inference_parameters(metadata)
    print(f"\ninferring {parameters}\n")

    for row in selected:
        out_hdf5 = os.path.join(out_dir, f"{row['name']}.hdf5")
        if os.path.exists(out_hdf5) and not args.force:
            print(f"[{row['name']}] exists -> {cc.rel(out_hdf5)}, skipping "
                  f"(--force to redo).", flush=True)
            continue

        # Same torch seed for every entry, so two posteriors differ through the
        # data and not through the flow's own sampling noise.
        torch.manual_seed(NOISE_SEED)
        torch.cuda.manual_seed_all(NOISE_SEED)

        t0 = time.time()
        result, pure_signal = inf._inject_and_infer(
            sampler, ig, theta1, NOISE_SEED, args.num_samples, batch_size,
            theta2=row["theta2"], shift_seconds=row["shift_seconds"],
            noise_seed=NOISE_SEED,
        )
        inf._save_result(result, pure_signal, out_hdf5, theta1, NOISE_SEED,
                         label, main_path, init_path,
                         theta2=row["theta2"], second_seed=NUISANCE_SEED,
                         shift_seconds=row["shift_seconds"])
        _tag_snrs(out_hdf5, snr1, snr1_det, row)
        print(f"[{row['name']}] {time.time() - t0:.1f}s -> {cc.rel(out_hdf5)}",
              flush=True)

    print(f"\nSeries in {cc.rel(out_dir)}")
    return out_dir


def _tag_snrs(out_hdf5, snr1, snr1_det, row):
    """Record the measured SNRs on the file.

    _build_injection only ever returns h1 + h2, so a reader cannot recover the
    per-signal SNRs from the stored strain; they have to be written down here.
    """
    import h5py

    with h5py.File(out_hdf5, "a") as h:
        h.attrs["snr1"] = float(snr1)
        for det, value in snr1_det.items():
            h.attrs[f"snr1_{det}"] = float(value)
        h.attrs["series_entry"] = row["name"]
        if row["theta2"] is not None:
            h.attrs["snr2"] = float(row["snr2"])
            for det, value in row["snr2_detectors"].items():
                h.attrs[f"snr2_{det}"] = float(value)
            h.attrs["snr_ratio"] = float(row["snr_ratio"])
            h.attrs["sky_separation_deg"] = float(row["sky_separation_deg"])
            h.attrs["source_separation_mpc"] = float(row["source_separation_mpc"])
            if row.get("axis"):
                h.attrs["series_axis"] = str(row["axis"])


if __name__ == "__main__":
    main()
