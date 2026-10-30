import glob
import os

import h5py
import numpy as np
import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_ROOT = os.path.dirname(HERE)
EVENTS_DIR = os.path.join(RESULT_ROOT, "events")        # observed strain (inputs)
LEGACY_DIR = os.path.join(RESULT_ROOT, "legacy")        # results from before the reorg
FIGURES_DIR = os.path.join(RESULT_ROOT, "figures")
TABLES_DIR = os.path.join(RESULT_ROOT, "tables")
INJECTIONS_DIR = os.path.join(LEGACY_DIR, "multi")


def list_injection_files(injections_dir: str = INJECTIONS_DIR) -> list[str]:
    files = sorted(glob.glob(os.path.join(injections_dir, "injection_*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No injection_*.hdf5 files found in {injections_dir}")
    return files


def load_injection_result(path: str):
    # Keep the lightweight diagnostic readers below usable without loading the
    # full Dingo/Bilby inference stack.
    from dingo.gw.result import Result

    return Result(file_name=path)


def load_injection_truth(path: str) -> dict[str, float]:
    """Read the exact parameters used to create an injection.

    These values are written with the result in ``/injection_parameters``.
    They are authoritative: reproducing a prior draw from a seed is unsafe,
    because the injection path uses both Bilby's and NumPy's random generators.
    """
    with h5py.File(path, "r") as f:
        if "injection_parameters" not in f:
            raise ValueError(
                f"{path} has no /injection_parameters group; cannot perform "
                "a truth-referenced diagnostic."
            )
        return {
            name: float(value[()])
            for name, value in f["injection_parameters"].items()
        }


def load_injection_posterior(path: str) -> pd.DataFrame:
    """Load posterior samples directly from the self-contained result HDF5."""
    with h5py.File(path, "r") as f:
        if "samples" not in f:
            raise ValueError(f"{path} has no /samples dataset.")
        return pd.DataFrame.from_records(f["samples"][()])


def get_posterior(result) -> pd.DataFrame:
    keys = [k for k in result.search_parameter_keys if k in result.samples.columns]
    return result.samples[keys].copy()


def read_injection(path: str) -> dict:
    result = load_injection_result(path)
    posterior = get_posterior(result)

    event_metadata = result.event_metadata or {}
    seed = event_metadata.get("injection_seed")

    basename = os.path.basename(path)
    index = int(os.path.splitext(basename)[0].split("_")[-1])

    return {
        "index": index,
        "path": path,
        "injection_seed": None if seed is None else int(seed),
        "keys": list(posterior.columns),
        "posterior": posterior,
    }


def load_all_injections(injections_dir: str = INJECTIONS_DIR) -> list[dict]:
    files = list_injection_files(injections_dir)
    records = [read_injection(f) for f in files]
    records.sort(key=lambda r: r["index"])
    return records


if __name__ == "__main__":
    records = load_all_injections()
    print(f"Loaded {len(records)} injection results from {INJECTIONS_DIR}")

    first = records[0]
    print(f"\nFirst injection: index={first['index']} seed={first['injection_seed']}")
    print(f"Inferred parameters ({len(first['keys'])}): {first['keys']}")
    print(f"Posterior samples per injection: {len(first['posterior'])}")

    n_samples = np.array([len(r["posterior"]) for r in records])
    seeds = [r["injection_seed"] for r in records]
    print(
        f"\nSample counts: min={n_samples.min()} max={n_samples.max()} "
        f"mean={n_samples.mean():.0f}"
    )
    print(f"Seeds present for all files: {all(s is not None for s in seeds)}")
