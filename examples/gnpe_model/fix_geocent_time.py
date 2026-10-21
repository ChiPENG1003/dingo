"""Relabel a waveform dataset's fiducial geocent_time to 0.

The ET dataset settings used `geocent_time: 0.1`, but dingo's frequency-domain
waveform generator never applies geocent_time to the polarizations (verified: the
h_plus/h_cross arrays are bit-identical for 0.0 and 0.1). The stored value is
therefore a mislabel, and ProjectOntoDetectors both asserts it is 0 and would
otherwise subtract it from every detector time (a 100 ms systematic shift).

This rewrites the `geocent_time` column and the corresponding entry in the stored
settings. It does not touch the polarizations. Reversible: pass --value 0.1.

Usage: python fix_geocent_time.py training_data/waveform_dataset_ET_5M.hdf5
"""

import argparse
import ast

import h5py
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in args.files:
        with h5py.File(path, "r+") as f:
            par = f["parameters"]
            before = np.unique(par["geocent_time"])
            settings = ast.literal_eval(f.attrs["settings"])
            stored = settings["intrinsic_prior"].get("geocent_time")
            print(f"{path}\n  column {before} (n={len(before)}), settings {stored!r}")
            if args.dry_run:
                continue

            data = par[:]                      # 5M x 12 float64 = 480 MB
            data["geocent_time"] = args.value
            par[...] = data

            settings["intrinsic_prior"]["geocent_time"] = args.value
            f.attrs["settings"] = str(settings)

            after = np.unique(f["parameters"]["geocent_time"])
            print(f"  -> column {after}, settings {args.value!r}")


if __name__ == "__main__":
    main()
