#!/usr/bin/env python
"""Check that a set of series files really is one signal 1 plus one variable.

Everything here is read back from the written files, not from the code that
wrote them, so it catches a stale file just as well as a bug:

  1. signal 1's truth is identical in every file;
  2. the noise realisation is identical in every file
     (noise = context/waveform - context/signal, both stored);
  3. pure_overlap - pure_reference is exactly the time-shifted signal 2 that the
     file says was injected (regenerated here from injection_parameters_2);
  4. the SNRs stored as attributes match a fresh measurement;
  5. the realised merger separation matches the shift the file claims;
  6. the sky angle and the 3-D source separation, rebuilt from the stored
     ra/dec (and distances), match the attributes -- `--backfill` writes them
     onto files that predate those attributes.

  python verify_series.py                       # every file in the series
  python verify_series.py 01_reference 11_dt0.05s 04_dt0.1s 02_dt0.5s

Needs no GPU: the checkpoint is opened for its metadata only.
"""

import argparse
import glob
import os

import h5py
import numpy as np

import combo_common as cc
import inference as inf
import run_series as rs

SERIES_AXIS = {name: (overrides.get("axis") or rs.axis_of(name))
               for name, _, overrides in rs.SERIES}

SERIES_DIR = os.path.join(cc.INFERENCE_DIR, "inference_result", "posteriors",
                          "1M_7params_O2", "series")
TOLERANCE = 1e-12


def read(path):
    with h5py.File(path, "r") as h:
        data = {det: np.asarray(h["context/waveform"][det]) for det in h["context/waveform"]}
        pure = {det: np.asarray(h["context/signal"][det]) for det in h["context/signal"]}
        asds = {det: np.asarray(h["context/asds"][det]) for det in h["context/asds"]}
        theta1 = {k: float(h["injection_parameters"][k][()])
                  for k in h["injection_parameters"]}
        theta2 = shift = None
        if "injection_parameters_2" in h:
            g2 = h["injection_parameters_2"]
            theta2 = {k: float(g2[k][()]) for k in g2}
            shift = float(g2.attrs["shift_seconds"])
        attrs = dict(h.attrs)
    noise = {det: data[det] - pure[det] for det in data}
    return dict(path=path, data=data, pure=pure, asds=asds, noise=noise,
                theta1=theta1, theta2=theta2, shift=shift, attrs=attrs)


def peak_time(h_freq, n, fs):
    return int(np.argmax(np.abs(np.fft.irfft(np.asarray(h_freq), n=n)))) / fs


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="*", default=None,
                   help="File stems in the series directory. Default: all of them.")
    p.add_argument("--series-dir", default=SERIES_DIR)
    p.add_argument("--model", default="1M_7params_O2")
    p.add_argument("--backfill", action="store_true",
                   help="Write the sky/source separation onto files that predate "
                        "those attributes, so every file can be placed on the "
                        "separation axis.")
    args = p.parse_args(argv)

    if args.names:
        paths = [os.path.join(args.series_dir, f"{n.removesuffix('.hdf5')}.hdf5")
                 for n in args.names]
    else:
        paths = sorted(glob.glob(os.path.join(args.series_dir, "*.hdf5")))
    files = [read(path) for path in paths]
    print(f"checking {len(files)} file(s) in {cc.rel(args.series_dir)}\n")

    _, main_path, _ = cc.resolve_model(args.model)
    ig = cc.injection_generator_from_metadata(cc.metadata_only(main_path))
    domain = ig.data_domain
    delta_f = domain.delta_f
    n_time = 2 * (len(files[0]["pure"][sorted(files[0]["pure"])[0]]) - 1)
    fs = n_time * delta_f

    reference = next((f for f in files if f["theta2"] is None), None)
    if reference is None:
        print("! no single-signal reference among these files; "
              "checks 2 and 3 are skipped.")

    ok = True

    # 1. one signal 1 everywhere
    base = files[0]["theta1"]
    for f in files[1:]:
        differing = {k: (base[k], f["theta1"][k]) for k in base
                     if not np.isclose(base[k], f["theta1"][k], rtol=0, atol=0)}
        if differing:
            ok = False
            print(f"!! {os.path.basename(f['path'])}: signal 1 differs: {differing}")
    print(f"1. signal 1 identical in all files: {'OK' if ok else 'FAILED'}")

    # 2. one noise realisation everywhere
    noise_ok = True
    for f in files[1:]:
        for det in files[0]["noise"]:
            scale = np.max(np.abs(files[0]["noise"][det]))
            diff = np.max(np.abs(files[0]["noise"][det] - f["noise"][det]))
            if diff > TOLERANCE * scale:
                noise_ok = ok = False
                print(f"!! {os.path.basename(f['path'])} {det}: noise differs by "
                      f"{diff:.3e} (scale {scale:.3e})")
    print(f"2. noise realisation identical in all files: "
          f"{'OK' if noise_ok else 'FAILED'}")

    # 3. the added signal is the one the file documents
    print("3. pure_overlap - pure_reference == shifted signal 2:")
    if reference is not None:
        for f in files:
            if f["theta2"] is None:
                continue
            waveform2 = ig.signal(f["theta2"])["waveform"]
            worst = 0.0
            for det in waveform2:
                expected = domain.update_data(
                    inf._linear_time_shift(domain, waveform2[det], f["shift"]),
                    low_value=0.0)
                got = f["pure"][det] - reference["pure"][det]
                worst = max(worst, np.max(np.abs(got - np.asarray(expected)))
                            / max(np.max(np.abs(expected)), 1e-300))
            flag = "OK" if worst < 1e-10 else "FAILED"
            if worst >= 1e-10:
                ok = False
            print(f"   {os.path.basename(f['path']):22s} relative residual "
                  f"{worst:.2e}  {flag}")

    # 4. stored SNRs vs a fresh measurement
    print("4. stored SNR attributes vs recomputed:")
    for f in files:
        snr1_stored = float(f["attrs"]["snr1"])
        snr1_now, _ = cc.network_snr(
            ig.signal(f["theta1"])["waveform"], f["asds"], delta_f)
        line = (f"   {os.path.basename(f['path']):22s} "
                f"SNR1 {snr1_stored:6.2f} vs {snr1_now:6.2f}")
        if f["theta2"] is not None:
            snr2_stored = float(f["attrs"]["snr2"])
            waveform2, _ = cc.projected_signal(ig, f["theta2"], f["shift"])
            snr2_now, _ = cc.network_snr(waveform2, f["asds"], delta_f)
            line += f" | SNR2 {snr2_stored:6.2f} vs {snr2_now:6.2f}"
            if abs(snr2_stored - snr2_now) > 1e-6 * snr2_now:
                ok = False
                line += "  !! mismatch"
        if abs(snr1_stored - snr1_now) > 1e-6 * snr1_now:
            ok = False
            line += "  !! SNR1 mismatch"
        print(line)

    # 5. the separation that was actually realised
    print("5. realised merger separation (geocentric shift + sky-position delay):")
    det0 = sorted(files[0]["pure"])[0]
    t1 = peak_time(ig.signal(files[0]["theta1"])["waveform"][det0], n_time, fs)
    period = n_time / fs
    for f in files:
        if f["theta2"] is None:
            print(f"   {os.path.basename(f['path']):22s} no second signal")
            continue
        shifted = inf._linear_time_shift(
            domain, ig.signal(f["theta2"])["waveform"][det0], f["shift"])
        t2 = peak_time(shifted, n_time, fs)
        dt = (t2 - t1 + period / 2) % period - period / 2
        print(f"   {os.path.basename(f['path']):22s} claims |dt| = {f['shift']:.3f} s, "
              f"measured {dt:+.4f} s in {det0}")

    # 6. the sky and source separations, rebuilt from the stored ra/dec
    print("6. sky / source separation vs the stored attributes:")
    for f in files:
        if f["theta2"] is None:
            continue
        t1, t2 = f["theta1"], f["theta2"]
        gamma = float(cc.sky_separation(t1["ra"], t1["dec"], t2["ra"], t2["dec"]))
        separation = float(cc.source_separation(
            t1["luminosity_distance"], t1["ra"], t1["dec"],
            t2["luminosity_distance"], t2["ra"], t2["dec"]))
        # Which sweep the entry belongs to, so a plot does not have to infer the
        # axis from which numbers happen to differ (rotating the sky moves ra and
        # dec too, and the inference picks those up).
        if args.backfill and "series_axis" not in f["attrs"]:
            stem = os.path.splitext(os.path.basename(f["path"]))[0]
            axis = SERIES_AXIS.get(stem)
            if axis:
                with h5py.File(f["path"], "a") as h:
                    h.attrs["series_axis"] = str(axis)

        stored_gamma = f["attrs"].get("sky_separation_deg")
        stored_sep = f["attrs"].get("source_separation_mpc")
        line = (f"   {os.path.basename(f['path']):22s} "
                f"gamma = {gamma:6.2f} deg, D = {separation:7.1f} Mpc")
        if stored_gamma is None:
            line += "   (not stored"
            line += "; --backfill writes it)" if not args.backfill else "; writing)"
            if args.backfill:
                with h5py.File(f["path"], "a") as h:
                    h.attrs["sky_separation_deg"] = gamma
                    h.attrs["source_separation_mpc"] = separation
        else:
            drift = max(abs(float(stored_gamma) - gamma),
                        abs(float(stored_sep) - separation) / max(separation, 1.0))
            if drift > 1e-6:
                ok = False
                line += f"   !! disagrees with the stored value by {drift:.3g}"
        print(line)

    print(f"\n{'ALL CHECKS PASSED' if ok else 'SOMETHING FAILED -- see above'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
