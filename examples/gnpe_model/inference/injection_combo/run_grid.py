"""Run one cell, one column, or a whole sweep of the overlapping-injection grid.

A sweep is a 10 x 10 grid: y is always the time separation, x is one of the axes
in grid_axes. Each cell holds `--num-realizations` injections that share the
cell's axis value and differ in their nuisance parameters and noise.

Signal 1 -- the target -- is identical in every injection of every sweep.

  python run_grid.py snr_ratio --cell 3 7            # one cell
  python run_grid.py chirp_mass                      # all 100 cells, serially
  python run_grid.py --reference                     # the no-distraction control

Under SLURM the array task id selects the cells, so submit_grid.sh only has to
say --array=0-99. Existing realisations are skipped, so a requeued or extended
job resumes instead of redoing work.
"""

import argparse
import os
import sys
import time

import h5py
import numpy as np
import torch

import combo_common as cc
import grid_axes as ga
import inference as inf
from select_target import build_signal1


def _add_args(p):
    p.add_argument("axis", nargs="?", default=None, choices=ga.AXIS_NAMES,
                   help="Which x axis to sweep. Omit with --reference.")
    p.add_argument("--reference", action="store_true",
                   help="Single-signal control: signal 1 alone, same noise seeds, "
                        "no grid. Gives the coverage the network reaches with no "
                        "distraction signal at all -- the baseline every heat map "
                        "has to be read against.")
    p.add_argument("--model", default=inf._env("MODEL", str, inf.DEFAULT_MODEL),
                   help="MODEL_REGISTRY key, or 'label=main.pt|init.pt'. Falls back "
                        "to $MODEL, then to inference.py's DEFAULT_MODEL.")
    p.add_argument("--target-seed", type=int, default=1,
                   help="Seed of signal 1. select_target.py reports which seeds "
                        "give a usable q_1 (it caps the chirp mass axis).")
    p.add_argument("--target-snr1", type=float, default=25.0,
                   help="Network SNR signal 1 is moved to. 0 keeps the prior draw.")
    p.add_argument("--num-realizations", type=int, default=100)
    p.add_argument("--num-bins", type=int, default=10)
    p.add_argument("--num-samples", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-iterations", type=int, default=30)
    p.add_argument("--cell", type=int, nargs=2, metavar=("X", "Y"), default=None)
    p.add_argument("--cells-per-task", type=int, default=1,
                   help="Cells handled by one SLURM array task. 10 turns the array "
                        "into one task per column.")
    p.add_argument("--sweep-name", default=None,
                   help="Output directory name. Defaults to the axis name.")
    p.add_argument("--ratio-min", type=float, default=0.2)
    p.add_argument("--ratio-max", type=float, default=5.0)
    p.add_argument("--distance-min", type=float, default=100.0)
    p.add_argument("--distance-max", type=float, default=2000.0)
    p.add_argument("--chirp-mass-min", type=float, default=10.0)
    p.add_argument("--chirp-mass-max", type=float, default=65.0)
    p.add_argument("--mass-ratio-min", type=float, default=0.2)
    p.add_argument("--mass-ratio-max", type=float, default=1.0)
    p.add_argument("--pinned-snr2", type=float, default=None,
                   help="SNR the distraction signal is pinned to on the mass axes. "
                        "Defaults to SNR_1, i.e. an equally loud intruder.")
    return p


def _axis_overrides(args):
    return {
        "ratio_min": args.ratio_min, "ratio_max": args.ratio_max,
        "distance_min": args.distance_min, "distance_max": args.distance_max,
        "chirp_mass_min": args.chirp_mass_min, "chirp_mass_max": args.chirp_mass_max,
        "mass_ratio_min": args.mass_ratio_min, "mass_ratio_max": args.mass_ratio_max,
        **({"pinned_snr2": args.pinned_snr2} if args.pinned_snr2 else {}),
    }


def _cells_for_this_task(args, num_x, num_y):
    """Which (x, y) cells this process is responsible for."""
    all_cells = [(x, y) for x in range(num_x) for y in range(num_y)]
    if args.cell is not None:
        return [tuple(args.cell)]
    task = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task is None:
        return all_cells
    start = int(task) * args.cells_per_task
    return all_cells[start:start + args.cells_per_task]


def _truth_quantiles(samples, theta, parameters):
    """Fraction of posterior mass below the truth, per parameter.

    This is the only thing the heat map needs: the truth is covered by the
    central C credible interval iff its quantile lies in ((1-C)/2, (1+C)/2), so
    the credible level can be changed at plotting time without re-running.
    """
    out = {}
    for name in parameters:
        if name not in samples:
            continue
        column = np.asarray(samples[name], dtype=float)
        out[name] = float(np.mean(column < float(theta[name])))
    return out


def _store_run(cell_file, index, theta2, quantiles, snr2, snr2_detectors, info,
               samples, parameters, elapsed):
    g = cell_file.create_group(f"run_{index:04d}")
    columns = [p for p in parameters if p in samples]
    g.create_dataset(
        "samples",
        data=np.asarray(samples[columns], dtype=np.float32),
        compression="gzip", compression_opts=4,
    )
    g["samples"].attrs["columns"] = columns
    for name, value in quantiles.items():
        g.attrs[f"quantile_{name}"] = value
    if theta2 is not None:
        for name, value in theta2.items():
            g.attrs[f"theta2_{name}"] = float(value)
        g.attrs["snr2"] = float(snr2)
        for det, value in snr2_detectors.items():
            g.attrs[f"snr2_{det}"] = float(value)
        if info.get("snr2_target") is not None:
            g.attrs["snr2_target"] = float(info["snr2_target"])
    g.attrs["seconds"] = float(elapsed)


def tag_of(axis, x_index, y_index, shift):
    if axis is None:
        return "reference"
    return (f"x{x_index:02d} y{y_index:02d}  {axis.name}="
            f"{axis.centers[x_index]:.4g}  dt={shift:.3f}s")


def run_cell(sampler, ctx, axis, x_index, y_index, args, parameters, sweep_name):
    """One cell: `--num-realizations` injections sharing an axis value."""
    shift = float(cc.TIME_SHIFTS[y_index]) if axis is not None else None
    path = (os.path.join(cc.REFERENCE_DIR, "reference.hdf5") if axis is None
            else cc.cell_path(sweep_name, x_index, y_index))
    os.makedirs(os.path.dirname(path), exist_ok=True)

    delta_f = ctx.ig.data_domain.delta_f
    batch_size = args.batch_size or args.num_samples
    done = skipped = 0
    failed = []
    t_cell = time.time()

    with h5py.File(path, "a") as f:
        f.attrs["sweep"] = sweep_name
        f.attrs["axis"] = "none" if axis is None else axis.name
        f.attrs["x_index"] = x_index
        f.attrs["y_index"] = y_index
        f.attrs["x_value"] = float("nan") if axis is None else float(axis.centers[x_index])
        f.attrs["shift_seconds"] = float("nan") if shift is None else shift
        f.attrs["snr1"] = ctx.snr1
        f.attrs["num_samples"] = args.num_samples
        if "theta1" not in f:
            g1 = f.create_group("theta1")
            for name, value in ctx.theta1.items():
                g1.attrs[name] = float(value)

        for k in range(args.num_realizations):
            if f"run_{k:04d}" in f:
                skipped += 1
                continue

            theta2 = None
            info = {}
            snr2, snr2_detectors = float("nan"), {}
            if axis is not None:
                theta2, info = ga.build_theta2(ctx, axis, x_index, shift, k)
                # Measure SNR_2 on the signal as actually injected -- projected
                # AND time-shifted. _build_injection only ever hands back the
                # *sum* h1 + h2, so the two SNRs have to be taken here.
                waveform2, asds = cc.projected_signal(ctx.ig, theta2, shift)
                snr2, snr2_detectors = cc.network_snr(waveform2, asds, delta_f)

            noise_seed = ctx.noise_seed(
                k + (axis.replica_offset(x_index) if axis is not None else 0))
            # Reseed torch so the flow draws the same base samples in every cell:
            # any difference between two cells at the same realisation index is
            # then attributable to the data, not to the sampler's own randomness.
            torch.manual_seed(noise_seed)
            torch.cuda.manual_seed_all(noise_seed)

            t0 = time.time()
            try:
                result, _ = inf._inject_and_infer(
                    sampler, ctx.ig, ctx.theta1, noise_seed,
                    args.num_samples, batch_size,
                    theta2=theta2, shift_seconds=shift, noise_seed=noise_seed,
                )
            except Exception as e:
                # One realisation must not cost the other 99. The usual culprit
                # is a numerical degeneracy inside the flow's rational-quadratic
                # spline (glasflow asserts `discriminant >= 0`), triggered when a
                # heavily contaminated strain drives the embedding somewhere the
                # network was never trained.
                #
                # It is NOT deterministic in the seed: the spline solve is
                # float32 on the GPU and the assertion is borderline, so kernel
                # selection decides it. Cell x02_y01 of the chirp_mass sweep died
                # here and then completed 100/100 clean on resubmission with
                # identical seeds. So a failed cell is worth simply re-running --
                # this skip exists to stop one bad draw taking the other 99 runs
                # of the task down with it, not because the draw is unfixable.
                failed.append(k)
                print(f"[{tag_of(axis, x_index, y_index, shift)}] realisation "
                      f"{k} FAILED, skipping: {type(e).__name__}: {e}", flush=True)
                continue
            elapsed = time.time() - t0

            quantiles = _truth_quantiles(result.samples, ctx.theta1, parameters)
            _store_run(f, k, theta2, quantiles, snr2, snr2_detectors, info,
                       result.samples, parameters, elapsed)
            f.flush()
            done += 1

    if failed:
        # Recorded on the cell, so a later reader can tell "this realisation is
        # missing because the flow blew up" from "this task was preempted".
        with h5py.File(path, "a") as f:
            previous = list(f.attrs.get("failed_realizations", []))
            f.attrs["failed_realizations"] = sorted(set(previous) | set(failed))

    tag = tag_of(axis, x_index, y_index, shift)
    print(f"[{tag}]  {done} new, {skipped} existing, {len(failed)} failed, "
          f"{time.time() - t_cell:.1f}s -> {cc.rel(path)}", flush=True)
    return path


def main(argv=None):
    args = _add_args(argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)).parse_args(argv)

    if args.axis is None and not args.reference:
        raise SystemExit("Give an axis name, or --reference. "
                         f"Axes: {', '.join(ga.AXIS_NAMES)}")

    label, main_path, init_path = cc.resolve_model(args.model)
    main_pm, init_pm = inf._load_pair(main_path, init_path)
    ig = inf.build_injection_generator(main_pm)
    parameters = cc.inference_parameters(main_pm.metadata)

    theta1, snr1, snr1_detectors = build_signal1(ig, args.target_seed, args.target_snr1)
    ctx = ga.Context(ig, theta1, snr1, snr1_detectors)

    axis = None
    sweep_name = "reference"
    if not args.reference:
        axis = ga.build_axis(args.axis, ctx, num_bins=args.num_bins,
                             **_axis_overrides(args))
        sweep_name = args.sweep_name or args.axis

    num_x = 1 if axis is None else len(axis.centers)
    num_y = 1 if axis is None else len(cc.TIME_SHIFTS)
    cells = [(0, 0)] if axis is None else _cells_for_this_task(args, num_x, num_y)

    inf._announce(
        f"grid:{sweep_name}", label,
        {"main checkpoint": cc.rel(main_path),
         "signal 1": f"seed {args.target_seed}, Mc {theta1['chirp_mass']:.2f}, "
                     f"q {theta1['mass_ratio']:.3f}, "
                     f"d {theta1['luminosity_distance']:.1f} Mpc",
         "SNR_1": f"{snr1:.3f}  (" +
                  ", ".join(f"{d} {v:.2f}" for d, v in sorted(snr1_detectors.items())) + ")",
         "x axis": "none (reference)" if axis is None else
                   f"{axis.name}: " + ", ".join(f"{c:.4g}" for c in axis.centers),
         "y axis": "none (reference)" if axis is None else
                   "dt = " + ", ".join(f"{t:.3f}" for t in cc.TIME_SHIFTS) + " s",
         "cells here": f"{len(cells)} x {args.num_realizations} realisations",
         "posterior": f"{args.num_samples} samples, "
                      f"{args.num_iterations} GNPE iterations"},
        cc.sweep_dir(sweep_name) if axis is not None else cc.REFERENCE_DIR,
        [f"cells/x{x:02d}_y{y:02d}.hdf5" for x, y in cells[:3]] +
        (["..."] if len(cells) > 3 else []),
    )

    # The sweep-wide record, written by whichever task gets there first. It is
    # what summarize_grid.py reads to label the axes, so it is never
    # reconstructed by hand later.
    config_path = os.path.join(cc.sweep_dir(sweep_name), "config.json")
    if not os.path.exists(config_path):
        cc.write_json(config_path, {
            "sweep": sweep_name,
            "axis": "none" if axis is None else axis.name,
            "axis_label": "" if axis is None else axis.label,
            "axis_centers": [] if axis is None else axis.centers.tolist(),
            "axis_edges": [] if axis is None else axis.edges.tolist(),
            "axis_notes": [] if axis is None else axis.notes,
            "time_shifts": cc.TIME_SHIFTS.tolist(),
            "shift_convention": "signal 2 merges this many seconds EARLIER "
                                "than signal 1",
            "model": label,
            "model_main": cc.rel(main_path),
            "model_init": cc.rel(init_path),
            "inference_parameters": parameters,
            "target_seed": args.target_seed,
            "target_snr1": snr1,
            "snr1_detectors": snr1_detectors,
            "theta1": theta1,
            "num_realizations": args.num_realizations,
            "num_samples": args.num_samples,
            "num_iterations": args.num_iterations,
            "nuisance_base_seed": ctx.nuisance_base_seed,
            "noise_base_seed": ctx.noise_base_seed,
        })

    sampler = inf.build_sampler(main_pm, init_pm, num_iterations=args.num_iterations)

    t0 = time.time()
    for x_index, y_index in cells:
        run_cell(sampler, ctx, axis, x_index, y_index, args, parameters, sweep_name)
    print(f"\n{len(cells)} cell(s) in {time.time() - t0:.1f}s -> "
          f"{cc.rel(cc.sweep_dir(sweep_name))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
