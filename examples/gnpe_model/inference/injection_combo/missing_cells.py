"""Which cells of a sweep are unfinished, as a SLURM array spec you can resubmit.

Array tasks die for reasons that have nothing to do with the science -- on this
cluster the usual one is several tasks sharing a GPU until one hits CUDA OOM at
model load. Those cells are simply absent or short, and re-running fills them in
because run_grid.py skips realisations that already exist.

  python missing_cells.py snr_ratio
  sbatch --array=$(python missing_cells.py snr_ratio --spec) submit_grid.sh snr_ratio

The task id for cell (x, y) is x * len(time_shifts) + y, matching
run_grid._cells_for_this_task at the default --cells-per-task 1.
"""

import argparse
import os

import h5py

import combo_common as cc


def cell_counts(sweep_name, num_x=10, num_y=10):
    """(x, y) -> number of finished realisations, for every cell of the grid."""
    counts = {}
    for x in range(num_x):
        for y in range(num_y):
            path = cc.cell_path(sweep_name, x, y)
            if not os.path.exists(path):
                counts[(x, y)] = 0
                continue
            try:
                with h5py.File(path, "r") as f:
                    counts[(x, y)] = sum(1 for k in f if k.startswith("run_"))
            except OSError:
                # A task killed mid-write leaves an unreadable file; treat it as
                # empty so it gets redone rather than silently trusted.
                counts[(x, y)] = -1
    return counts


def _compress(task_ids):
    """[0,1,2,5,7,8] -> '0-2,5,7-8', the form sbatch --array wants."""
    if not task_ids:
        return ""
    task_ids = sorted(task_ids)
    parts, start, prev = [], task_ids[0], task_ids[0]
    for task in task_ids[1:] + [None]:
        if task is not None and task == prev + 1:
            prev = task
            continue
        parts.append(f"{start}" if start == prev else f"{start}-{prev}")
        start = prev = task
    return ",".join(parts)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("sweep")
    p.add_argument("--expected", type=int, default=None,
                   help="Realisations a complete cell holds. Defaults to the "
                        "sweep's own config.json.")
    p.add_argument("--spec", action="store_true",
                   help="Print only the array spec, for use inside $(...).")
    args = p.parse_args(argv)

    config_path = os.path.join(cc.sweep_dir(args.sweep), "config.json")
    expected = args.expected
    num_y = 10
    if os.path.exists(config_path):
        config = cc.read_json(config_path)
        expected = expected or config["num_realizations"]
        num_y = len(config["time_shifts"])
        num_x = len(config["axis_centers"]) or 10
    else:
        expected, num_x = expected or 100, 10

    counts = cell_counts(args.sweep, num_x=num_x, num_y=num_y)
    incomplete = {cell: n for cell, n in counts.items() if n < expected}
    task_ids = [x * num_y + y for (x, y) in incomplete]

    if args.spec:
        print(_compress(task_ids))
        return _compress(task_ids)

    done = len(counts) - len(incomplete)
    total_runs = sum(max(n, 0) for n in counts.values())
    print(f"sweep {args.sweep}: {done}/{len(counts)} cells complete, "
          f"{total_runs}/{len(counts) * expected} injections done")
    if not incomplete:
        print("nothing to resubmit.")
        return ""
    for (x, y), n in sorted(incomplete.items()):
        state = "unreadable" if n < 0 else f"{n}/{expected}"
        print(f"  x{x:02d} y{y:02d}  task {x * num_y + y:3d}  {state}")
    spec = _compress(task_ids)
    print(f"\nresubmit with:\n  sbatch --array={spec} submit_grid.sh {args.sweep}")
    return spec


if __name__ == "__main__":
    main()
