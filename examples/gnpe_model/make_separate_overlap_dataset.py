#!/usr/bin/env python
import argparse
import ast
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--signal1", required=True)
    p.add_argument("--signal2", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--suffix", default="_2", help="key suffix for signal 2")
    p.add_argument("--chunk-size", type=int, default=100000)
    p.add_argument("--num-samples", type=int, default=None,
                   help="truncate (for a quick end-to-end test)")
    p.add_argument("--check-samples", type=int, default=5000,
                   help="how many waveforms to decompress for the accuracy check")
    return p.parse_args()


def check_compatible(st1, st2, path1, path2):
    """The two inputs must be interchangeable apart from their SVD bases."""
    for key in ("domain", "waveform_generator"):
        if st1[key] != st2[key]:
            raise ValueError(
                f"{key} differs between inputs:\n  {path1}: {st1[key]}\n  {path2}: {st2[key]}"
            )
    w1 = st1["compression"].get("whitening")
    w2 = st2["compression"].get("whitening")
    if w1 != w2:
        raise ValueError(f"whitening ASD differs: {w1!r} vs {w2!r}")
    if st1["intrinsic_prior"] != st2["intrinsic_prior"]:
        raise ValueError(
            "intrinsic priors differ; the two signals must be drawn from the same "
            "population for the shared SVD basis to be valid."
        )


def main():
    args = parse_args()
    t0 = time.time()

    f1 = h5py.File(args.signal1, "r")
    f2 = h5py.File(args.signal2, "r")
    st1 = ast.literal_eval(f1.attrs["settings"])
    st2 = ast.literal_eval(f2.attrs["settings"])
    check_compatible(st1, st2, args.signal1, args.signal2)

    pols = sorted(f1["polarizations"].keys())
    if pols != sorted(f2["polarizations"].keys()):
        raise ValueError("polarizations differ between inputs")
    n1 = f1["polarizations"][pols[0]].shape[0]
    n2 = f2["polarizations"][pols[0]].shape[0]
    n_total = args.num_samples or min(n1, n2)
    if n1 != n2 and args.num_samples is None:
        print(f"WARNING: sample counts differ ({n1} vs {n2}); using {n_total}")

    V1 = f1["svd/V"][:]                       # (n_freq, k1)
    V2 = f2["svd/V"][:]                       # (n_freq, k2)
    k1, k2 = V1.shape[1], V2.shape[1]
    print(f"signal 1: {args.signal1}  ({n1} samples, SVD {k1})")
    print(f"signal 2: {args.signal2}  ({n2} samples, SVD {k2})")
    print(f"combining {n_total} samples, polarizations {pols}, suffix '{args.suffix}'")
    print(f"domain: {st1['domain']}")

    M = V2.conj().T @ V1                      # (k2, k1)

    m = min(args.check_samples, n_total)
    c2 = f2["polarizations"]["h_plus"][:m]
    h2 = c2 @ V2.conj().T
    h2_proj = (c2 @ M) @ V1.conj().T
    num = np.sum(np.abs(h2 - h2_proj) ** 2, axis=1)
    den = np.sum(np.abs(h2) ** 2, axis=1)
    inner = np.abs(np.sum(h2.conj() * h2_proj, axis=1))
    mm = 1 - inner / np.sqrt(den * np.sum(np.abs(h2_proj) ** 2, axis=1))
    print(f"\nsignal 2 re-expressed in signal 1's basis ({m} waveforms):")
    print(f"  residual power fraction: median {np.median(num/den):.3e}  max {np.max(num/den):.3e}")
    print(f"  mismatch:                median {np.median(mm):.3e}  max {np.max(mm):.3e}")
    del c2, h2, h2_proj

    tmp = args.out + ".partial"
    print(f"\nWriting {tmp}")
    with h5py.File(tmp, "w") as out:
        grp = out.create_group("polarizations")
        dsets = {}
        for pol in pols:
            dsets[pol] = grp.create_dataset(pol, shape=(n_total, k1), dtype=np.complex128)
            dsets[pol + args.suffix] = grp.create_dataset(
                pol + args.suffix, shape=(n_total, k1), dtype=np.complex128
            )
        for lo in range(0, n_total, args.chunk_size):
            hi = min(lo + args.chunk_size, n_total)
            for pol in pols:
                # signal 1 already lives in this basis: straight copy
                dsets[pol][lo:hi] = f1["polarizations"][pol][lo:hi]
                dsets[pol + args.suffix][lo:hi] = f2["polarizations"][pol][lo:hi] @ M
            el = time.time() - t0
            done = hi / n_total
            print(f"  {hi}/{n_total} ({100*done:.1f}%)  elapsed {el/60:.1f} min  "
                  f"eta {el*(1-done)/done/60:.1f} min", flush=True)

        p1 = pd.DataFrame(f1["parameters"][:n_total])
        p2 = pd.DataFrame(f2["parameters"][:n_total]).add_suffix(args.suffix)
        params = pd.concat([p1, p2], axis=1)
        out.create_dataset("parameters", data=params.to_records(index=False))

        svd = out.create_group("svd")
        svd.create_dataset("V", data=V1)
        svd.create_dataset("s", data=f1["svd/s"][:])
        if "mismatches" in f1["svd"]:
            svd.create_dataset("mismatches", data=f1["svd/mismatches"][:])

        settings = dict(st1)
        settings["num_samples"] = n_total
        settings["overlap"] = {
            "layout": "signals stored separately; summed after projection at train time",
            "signal1": os.path.basename(args.signal1),
            "signal2": os.path.basename(args.signal2),
            "suffix": args.suffix,
            "shared_basis": "signal 1",
            "signal2_reprojection_mismatch_median": float(np.median(mm)),
            "signal2_reprojection_mismatch_max": float(np.max(mm)),
            "note": "time offset, amplitude ratio and sky positions are sampled at "
                    "train time; see dingo.gw.transforms.overlap_transforms",
        }
        out.attrs["settings"] = str(settings)
        out.attrs["dataset_type"] = "waveform_dataset"
        out.create_dataset("version", data=f"separate-overlap-{np.datetime64('now')}")

    os.replace(tmp, args.out)
    f1.close()
    f2.close()
    print(f"\nDone: {args.out} ({os.path.getsize(args.out)/1e9:.1f} GB) "
          f"in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
