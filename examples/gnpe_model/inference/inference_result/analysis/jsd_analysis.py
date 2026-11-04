import os
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_PATH = (
    RESULT_ROOT/"posteriors"/"ET_complete"/"single"
)
FILE_PATH = (
    RESULT_ROOT/"posteriors"/"ET_identical_dt_seed10073"/"overlap"
)

BINS = 30


def load_truth(path):
    with h5py.File(path, "r") as f:
        if "injection_parameters" not in f:
            raise KeyError(f"{path}: missing injection_parameters")

        return {
            name: float(value[()])
            for name, value in f["injection_parameters"].items()
        }


def load_samples(path):
    with h5py.File(path, "r") as f:
        if "samples" not in f:
            raise KeyError(f"{path}: missing samples")

        samples = f["samples"]

        if isinstance(samples, h5py.Group):
            return {
                name: np.asarray(values[()], dtype=float)
                for name, values in samples.items()
            }

        data = samples[()]
        if data.dtype.names:
            return {
                name: np.asarray(data[name], dtype=float)
                for name in data.dtype.names
            }

        raise ValueError(
            f"{path}: unsupported samples format; "
            f"shape={data.shape}, dtype={data.dtype}"
        )


def read_file(path=FILE_PATH, seed=10073):
    result_files = sorted(
        Path(path).glob(f"seed{seed}_*_identical_logdt_*.hdf5")
    )

    if not result_files:
        injection_file = Path(path) / f"seed{seed}_dprior.hdf5"
        if not injection_file.is_file():
            raise FileNotFoundError(
                f"No result files found for seed {seed} in {path}"
            )
        print("loading reference injection...")
        return {
            0: {
                "file": injection_file,
                "time_shift": 0.0,
                "filename_shift": 0.0,
                "truth": load_truth(injection_file),
                "samples": load_samples(injection_file),
            }
        }

    injections = {}

    for result_file in result_files:
        parts = result_file.stem.split("_")

        shift_part = next(
            part for part in parts if part.startswith("shift")
        )
        filename_shift = float(
            shift_part.removeprefix("shift").removesuffix("s")
        )

        index = int(parts[-1])

        with h5py.File(result_file, "r") as f:
            group = f.get("injection_parameters_2")

            if group is not None and "shift_seconds" in group.attrs:
                time_shift = float(group.attrs["shift_seconds"])
            else:
                if not 0 <= index < 20:
                    raise ValueError(f"Unexpected index: {index}")
                time_shift = float(np.geomspace(0.01, 1.0, 20)[index])

        if index in injections:
            raise ValueError(f"Duplicate injection index: {index}")

        injections[index] = {
            "file": result_file,
            "time_shift": time_shift,
            "filename_shift": filename_shift,
            "truth": load_truth(result_file),
            "samples": load_samples(result_file),
        }
    return dict(sorted(injections.items()))


def get_90ci(samples):
    samples = np.asarray(samples, dtype =float)
    samples = samples[np.isfinite(samples)]
    lo, hi = np.quantile(samples, [0.05, 0.95])
    print(f"90CI: [{lo}, {hi}]")

if __name__ == "__main__":
    injections = read_file(path=FILE_PATH, seed=10073)
    reference_injection = read_file(path=REFERENCE_PATH, seed=10073)
    samples = injections[3]["samples"]["chirp_mass"]
    ref_truth = reference_injection[0]["truth"]["chirp_mass"]
    truth = injections[3]["truth"]["chirp_mass"]
    time_shift = injections[3]["time_shift"]
    print(ref_truth)
    print(truth, time_shift)
    get_90ci(samples)
    # for index, injection in injections.items():
    #     print(
    #         f"index={index:02d}, "
    #         f"time shift={injection['time_shift']:.8f} s, "
    #         f"parameters={list(injection['samples'])}"
    #     )