#!/usr/bin/env python

import argparse
import ast
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd

from dingo.gw.SVD import SVDBasis


# ---------------------------------------------------------------------------
# input handling
# ---------------------------------------------------------------------------


class Source:
    """One input dataset: its SVD basis, coefficients, and parameters."""

    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        self.settings = ast.literal_eval(self.f.attrs["settings"])
        self.Vh = self.f["svd/V"][:].T.conj()
        self.pols = list(self.f["polarizations"].keys())
        self.n_samples = self.f["polarizations"][self.pols[0]].shape[0]

    def coefficients(self, pol, lo, hi):
        return self.f["polarizations"][pol][lo:hi]

    def waveforms(self, pol, lo, hi):
        """Whitened frequency-domain polarizations, SVD-decompressed."""
        return self.coefficients(pol, lo, hi) @ self.Vh

    def parameters(self, lo, hi):
        return pd.DataFrame(self.f["parameters"][lo:hi])

    def close(self):
        self.f.close()


def check_compatible(s1, s2):
    """The two inputs must be interchangeable apart from their SVD bases."""
    for key in ("domain", "waveform_generator"):
        if s1.settings[key] != s2.settings[key]:
            raise ValueError(
                f"{key} differs between inputs:\n"
                f"  {s1.path}: {s1.settings[key]}\n"
                f"  {s2.path}: {s2.settings[key]}"
            )
    w1 = s1.settings["compression"].get("whitening")
    w2 = s2.settings["compression"].get("whitening")
    if w1 != w2:
        raise ValueError(f"whitening ASD differs: {w1!r} vs {w2!r}")
    if s1.pols != s2.pols:
        raise ValueError(f"polarizations differ: {s1.pols} vs {s2.pols}")
    if s1.n_samples != s2.n_samples:
        raise ValueError(
            f"sample counts differ: {s1.n_samples} vs {s2.n_samples}. Combining "
            f"would silently truncate to the smaller one."
        )


# ---------------------------------------------------------------------------
# the overlap itself
# ---------------------------------------------------------------------------


def shift_earlier(h, frequencies, shifts):
    """Move each waveform earlier in time by shifts[i] seconds.

    dingo's domain.time_translate_data(data, dt) multiplies by exp(-2 pi i f dt),
    so moving to t = -n means dt = -n, i.e. a factor exp(+2 pi i f n).
    """
    return h * np.exp(2j * np.pi * frequencies[None, :] * shifts[:, None])


def joint_waveforms(s1, s2, pol, lo, hi, frequencies, shifts):
    return s1.waveforms(pol, lo, hi) + shift_earlier(
        s2.waveforms(pol, lo, hi), frequencies, shifts[lo:hi]
    )


def check_wrap(joint, delta_f, label="", guard_seconds=0.5):
    """Report how much signal energy sits near the circular-shift wrap point.

    In the irfft buffer, index j maps to t = j/fs for j <= n/2 and to
    t = (j-n)/fs above it: the merger sits just after j = 0 and the inspiral
    runs backwards from the top of the buffer. The two ends therefore meet at
    j = n/2, i.e. t = +/- T/2 = +/- 4 s -- that, not the buffer end, is where a
    circular shift folds signal back onto itself. A clean shift leaves this
    band empty.
    """
    n_time = 2 * (joint.shape[1] - 1)
    fs = n_time * delta_f
    ht = np.fft.irfft(joint, n=n_time, axis=1)
    power = ht.real**2
    total = power.sum(axis=1)
    guard = max(1, int(round(guard_seconds * fs)))
    mid = n_time // 2
    band = power[:, mid - guard : mid + guard].sum(axis=1)
    frac = np.where(total > 0, band / np.maximum(total, 1e-300), 0.0)
    print(
        f"  wrap check{label}: fraction of signal energy within +/-{guard_seconds}s "
        f"of the t=+/-{1/(2*delta_f)}s wrap point -- "
        f"median {np.median(frac):.3e}, max {frac.max():.3e}"
    )
    return frac.max()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--signal1", required=True, help="dataset kept at t=0")
    p.add_argument("--signal2", required=True, help="dataset shifted earlier")
    p.add_argument("--out", required=True, help="output joint dataset")
    p.add_argument("--shift-min", type=float, default=0.01, help="seconds (default 0.01)")
    p.add_argument("--shift-max", type=float, default=0.5, help="seconds (default 0.5)")
    p.add_argument("--svd-size", type=int, default=600)
    p.add_argument("--svd-train", type=int, default=20000)
    p.add_argument("--svd-test", type=int, default=5000)
    p.add_argument("--chunk-size", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=None,
                   help="truncate (for a quick end-to-end test)")
    p.add_argument("--check-wrap", action="store_true",
                   help="verify the circular shift does not wrap signal energy")
    return p.parse_args()


def main():
    args = parse_args()
    t_start = time.time()

    s1, s2 = Source(args.signal1), Source(args.signal2)
    check_compatible(s1, s2)

    n_total = args.num_samples or s1.n_samples
    pols = s1.pols
    delta_f = s1.settings["domain"]["delta_f"]
    n_freq = s1.Vh.shape[1]
    frequencies = np.arange(n_freq) * delta_f
    min_idx = int(round(s1.settings["domain"]["f_min"] / delta_f))

    print(f"signal 1: {args.signal1}  ({s1.n_samples} samples)")
    print(f"signal 2: {args.signal2}  ({s2.n_samples} samples)")
    print(f"combining {n_total} samples, polarizations {pols}")
    print(f"domain: {n_freq} bins, delta_f={delta_f}, T={1/delta_f}s, f_min index {min_idx}")
    print(f"shift: signal 2 earlier by U({args.shift_min}, {args.shift_max}) s")

    if args.shift_max >= 1 / delta_f:
        raise ValueError(
            f"shift_max {args.shift_max}s exceeds the buffer length {1/delta_f}s; "
            f"a circular shift would wrap the signal onto itself."
        )

    rng = np.random.default_rng(args.seed)
    shifts = rng.uniform(args.shift_min, args.shift_max, size=n_total)

    # --- 1. build the joint SVD basis -------------------------------------
    n_basis = min(args.svd_train + args.svd_test, n_total)
    n_train = int(round(args.svd_train / (args.svd_train + args.svd_test) * n_basis))
    print(f"\nBuilding joint SVD basis from {n_train} training / "
          f"{n_basis - n_train} validation waveforms per polarization.")

    basis_data = {
        pol: joint_waveforms(s1, s2, pol, 0, n_basis, frequencies, shifts)
        for pol in pols
    }
    if args.check_wrap:
        for pol in pols:
            check_wrap(basis_data[pol][:2000], delta_f, label=f" [{pol}]")

    train_data = np.vstack([basis_data[pol][:n_train] for pol in pols])
    basis = SVDBasis()
    basis.generate_basis(train_data, args.svd_size)
    del train_data

    if not np.allclose(basis.V[:min_idx], 0):
        raise ValueError("SVD basis is nonzero below f_min; inputs are inconsistent.")

    test_data = np.vstack([basis_data[pol][n_train:] for pol in pols])
    if test_data.size:
        test_params = pd.concat(
            [s1.parameters(n_train, n_basis) for _ in pols], ignore_index=True
        )
        basis.compute_test_mismatches(
            test_data, parameters=test_params, increment=100, verbose=True
        )
    del test_data, basis_data

    V = basis.V
    print(f"basis built in {time.time() - t_start:.0f}s")

    # --- 2. stream through the dataset ------------------------------------
    tmp_out = args.out + ".partial"
    print(f"\nWriting {tmp_out}")
    with h5py.File(tmp_out, "w") as out:
        grp = out.create_group("polarizations")
        dsets = {
            pol: grp.create_dataset(pol, shape=(n_total, args.svd_size),
                                    dtype=np.complex128)
            for pol in pols
        }
        for lo in range(0, n_total, args.chunk_size):
            hi = min(lo + args.chunk_size, n_total)
            for pol in pols:
                joint = joint_waveforms(s1, s2, pol, lo, hi, frequencies, shifts)
                dsets[pol][lo:hi] = joint @ V
            elapsed = time.time() - t_start
            done = hi / n_total
            print(f"  {hi}/{n_total} ({100*done:.1f}%)  "
                  f"elapsed {elapsed/60:.1f} min  eta {elapsed*(1-done)/done/60:.1f} min",
                  flush=True)

        # --- 3. parameters: signal 1 under its own names, signal 2 suffixed
        p1 = s1.parameters(0, n_total)
        p2 = s2.parameters(0, n_total).add_suffix("_2")
        params = pd.concat([p1, p2], axis=1)
        params["delta_tc_2"] = shifts   # seconds signal 2 was moved earlier
        out.create_dataset("parameters", data=params.to_records(index=False))

        # --- 4. SVD basis and settings ------------------------------------
        svd = out.create_group("svd")
        svd.create_dataset("V", data=basis.V)
        svd.create_dataset("s", data=basis.s)
        if basis.mismatches is not None:
            svd.create_dataset("mismatches",
                               data=basis.mismatches.to_records(index=False))

        settings = dict(s1.settings)
        settings["num_samples"] = n_total
        settings["compression"] = dict(s1.settings["compression"])
        settings["compression"]["svd"] = {
            "size": args.svd_size,
            "num_training_samples": n_train,
            "num_validation_samples": n_basis - n_train,
        }
        settings["overlap"] = {
            "signal1": os.path.basename(args.signal1),
            "signal2": os.path.basename(args.signal2),
            "shift_min": args.shift_min,
            "shift_max": args.shift_max,
            "shift_parameter": "delta_tc_2",
            "shift_convention": "signal 2 merger at t = -delta_tc_2; signal 1 at t = 0",
            "seed": args.seed,
        }
        out.attrs["settings"] = str(settings)
        out.attrs["dataset_type"] = "waveform_dataset"
        out.create_dataset("version", data=f"overlap-{np.datetime64('now')}")

    os.replace(tmp_out, args.out)
    s1.close()
    s2.close()
    size_gb = os.path.getsize(args.out) / 1e9
    print(f"\nDone: {args.out} ({size_gb:.1f} GB) in {(time.time()-t_start)/60:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
