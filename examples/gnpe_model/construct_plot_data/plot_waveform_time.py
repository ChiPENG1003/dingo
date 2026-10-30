"""Time-domain (time vs. amplitude) plot of any waveform signal.

Two sources:

  dataset    -- pick one or more samples from a waveform dataset by index (or at
                random), give each its own merger time offset / amplitude scale,
                and plot the SUM as a single trace. Overlapping signals are made
                this way; the individual signals are not separated.

  injection  -- read an injection HDF5 written by inference.py and plot its
                time-domain strain. Overlap injections (MODE=overlap) already
                contain both signals summed in context/signal.

Examples
--------
  # two overlapping dataset waveforms, second one merging 0.15 s later
  python plot_waveform_time.py --index 12 3400 --shift 0 0.15

  # three random signals, whitened (the chirps become visible)
  python plot_waveform_time.py --random 3 --whiten

  # a stored injection (pure signal), both detectors
  python plot_waveform_time.py --injection ../inference/inference_result/legacy/multi/injection_0000.hdf5

  # the same injection as it is actually seen by the network: h + n, whitened
  python plot_waveform_time.py --injection ... --part data --whiten --tmin -1
"""

import argparse
import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from construct_data import (
    DELTA_F,
    F_MAX,
    F_MIN,
    freq_to_time_domain,
    frequency_domain_asd_whiten,
    get_detector_asd_data,
    get_fixed_whitening_asd,
    read_injection,
    EXAMPLE_DIR,
    TRAINING_DATA_DIR,
    RESULT_ROOT,
)

DEFAULT_DATASET = os.path.join(TRAINING_DATA_DIR, "waveform_dataset_500K_SVD300.hdf5")
OUT_DIR = Path(EXAMPLE_DIR) / "picture" / "time_domain"


def load_waveform_freq(dataset_path, sample_idx, polarization="h_cross"):
    """Frequency-domain polarization of one dataset sample (SVD-decompressed,
    de-whitened back to physical strain) plus its parameters."""
    with h5py.File(dataset_path, "r") as f:
        raw = f[f"polarizations/{polarization}"][sample_idx : sample_idx + 1]
        has_svd = "svd" in f and "V" in f["svd"]
        V = f["svd/V"][:] if has_svd else None
        row = f["parameters"][sample_idx]
        params = {name: float(row[name]) for name in row.dtype.names}

    h_freq = (raw @ V.conj().T if has_svd else raw)[0]
    freq = np.arange(len(h_freq)) * DELTA_F
    # The dataset stores waveforms whitened by a fixed reference ASD (compression);
    # undo that to get physical strain.
    h_freq = h_freq * get_fixed_whitening_asd(freq)
    return freq, h_freq, params


def shift_time_series(x, seconds, fs):
    """Shift x later in time by `seconds` (negative = earlier), zero-filling.

    Not circular: whatever is pushed out of the segment is dropped, so a large
    shift simply moves the signal out of the plotted window.
    """
    k = int(round(seconds * fs))
    out = np.zeros_like(x)
    n = len(x)
    if k == 0:
        return x.copy()
    if abs(k) >= n:
        return out
    if k > 0:
        out[k:] = x[: n - k]
    else:
        out[: n + k] = x[-k:]
    return out


def whiten_with_asd(x_time, fs, asd_freq, asd, fmin=F_MIN, fmax=F_MAX):
    """Whiten x_time by an ASD given on an arbitrary frequency grid."""
    freq = np.fft.rfftfreq(len(x_time), d=1.0 / fs)
    asd_interp = np.interp(freq, asd_freq, asd, left=asd[0], right=asd[-1])
    return frequency_domain_asd_whiten(x_time, asd_interp, fs, f_min=fmin, f_max=fmax)


def build_dataset_signal(
    dataset_path,
    indices,
    shifts,
    scales,
    polarization="h_cross",
    t_window=8.0,
    end_margin=0.1,
):
    """Sum the selected dataset waveforms into one time series.

    Returns (time, total, fs, info) with t = 0 at the merger of the FIRST
    (unshifted) signal, `total` the summed strain and `info` a per-signal list
    of (index, params, shift, scale, component time series).
    """
    total = None
    fs = None
    info = []
    for idx, shift, scale in zip(indices, shifts, scales):
        freq, h_freq, params = load_waveform_freq(dataset_path, idx, polarization)
        h_time, fs, _ = freq_to_time_domain(
            h_freq,
            freq,
            params["chirp_mass"],
            t_window=t_window,
            t_end_margin=end_margin,
        )
        component = scale * shift_time_series(h_time, shift, fs)
        total = component if total is None else total + component
        info.append((idx, params, shift, scale, component))

    time = np.arange(len(total)) / fs - (t_window - end_margin)
    return time, total, fs, info


def plot_dataset_signal(args):
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.dataset, "r") as f:
        n_samples = f[f"polarizations/{args.polarization}"].shape[0]

    if args.index:
        indices = list(args.index)
    else:
        indices = [int(i) for i in rng.integers(0, n_samples, size=args.random)]

    shifts = list(args.shift) + [0.0] * (len(indices) - len(args.shift))
    scales = list(args.scale) + [1.0] * (len(indices) - len(args.scale))
    if len(shifts) > len(indices) or len(scales) > len(indices):
        raise SystemExit("--shift / --scale must not have more entries than signals.")

    # Signals are placed with their merger `end_margin` seconds before the end of
    # the segment, so a positive shift would push a merger out of it. Make room
    # for the latest one.
    end_margin = args.end_margin + max(0.0, max(shifts))

    time, total, fs, info = build_dataset_signal(
        args.dataset,
        indices,
        shifts,
        scales,
        polarization=args.polarization,
        t_window=args.t_window,
        end_margin=end_margin,
    )

    label = "whitened strain" if args.whiten else r"$h(t)$"
    if args.whiten:
        asd_freq, _, asd = get_detector_asd_data(args.observing_run, args.detector[0])
        total = whiten_with_asd(total, fs, asd_freq, asd)
        components = [
            whiten_with_asd(c, fs, asd_freq, asd) for (_, _, _, _, c) in info
        ]
    else:
        components = [c for (_, _, _, _, c) in info]

    show_components = len(info) > 1 and not args.no_components

    fig, ax = plt.subplots(figsize=(12, 4.5))
    if show_components:
        # Single signals and their sum are drawn identically (same width, default
        # colour cycle), so the sum is just one more line rather than a trace that
        # visually dominates the others.
        for (idx, _, _, _, _), comp in zip(info, components):
            ax.plot(time, comp, lw=0.9, label=f"sample {idx}")
        ax.plot(time, total, lw=0.9, label="sum")
    else:
        ax.plot(time, total, lw=0.8, color="black",
                label="sum" if len(info) > 1 else f"sample {info[0][0]}")

    lines = []
    for idx, params, shift, scale, _ in info:
        Mc, q = params["chirp_mass"], params["mass_ratio"]
        m1 = Mc * (1 + q) ** 0.2 / q ** 0.6
        lines.append(
            f"[{idx}] Mc={Mc:.1f}, m=({m1:.1f},{m1 * q:.1f}), "
            f"d_L={params['luminosity_distance']:.0f} Mpc, "
            f"dt={shift:+.3f} s, x{scale:g}"
        )
    ax.set_title(
        f"{len(info)} signal(s), {args.polarization}"
        + (f", whitened ({args.observing_run} {args.detector[0]} ASD)" if args.whiten else "")
    )
    ax.text(
        0.01, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", ha="left",
        fontsize=7, family="monospace",
        bbox=dict(boxstyle="round", fc="white", alpha=0.7, lw=0.5),
    )
    ax.set_xlabel("Time relative to first merger [s]")
    ax.set_ylabel(label)
    ax.set_xlim(args.tmin if args.tmin is not None else time[0],
                args.tmax if args.tmax is not None else time[-1])
    if show_components:
        ax.legend(loc="lower left", fontsize=7, ncol=len(info) + 1)
    ax.grid(alpha=0.2)

    out_name = args.out or ("waveform_time_" + "_".join(str(i) for i in indices) + ".png")
    return fig, out_name


def plot_injection_signal(args):
    data, asds, signals = read_injection(args.injection)

    if args.part == "signal":
        if not signals:
            print("[WARN] no context/signal in this file; falling back to d = h + n.")
            source = data
        else:
            source = signals
    else:
        source = data

    detectors = [d for d in args.detector if d in source] or list(source)

    fig, axes = plt.subplots(
        len(detectors), 1, figsize=(12, 3.0 * len(detectors)), sharex=True, squeeze=False
    )
    for ax, det in zip(axes[:, 0], detectors):
        x_freq = source[det]
        n_time = 2 * (len(x_freq) - 1)
        fs = n_time * DELTA_F
        x_time = fs * np.fft.irfft(x_freq, n=n_time)

        if args.whiten:
            x_time = frequency_domain_asd_whiten(x_time, asds[det], fs)

        # Injection buffers are cyclic: anchor t = 0 on the primary merger, i.e.
        # the strong sample whose wrapped buffer time is closest to zero (not the
        # global-loudest, which may be the second, louder signal of an overlap).
        env = np.abs(x_time if args.whiten
                     else frequency_domain_asd_whiten(x_time, asds[det], fs))
        tau = np.arange(n_time) / fs
        tau = np.where(tau >= (n_time / fs) / 2.0, tau - n_time / fs, tau)
        strong = np.where(env > 0.25 * env.max())[0]
        merger_idx = int(strong[np.argmin(np.abs(tau[strong]))]) if len(strong) else 0

        target_idx = n_time - 1 - int(round(args.end_margin * fs))
        x_time = np.roll(x_time, target_idx - merger_idx)
        time = np.arange(n_time) / fs - target_idx / fs

        ax.plot(time, x_time, lw=0.6, color="black")
        ax.set_ylabel(f"{det}\n" + ("whitened" if args.whiten else r"$h(t)$"))
        ax.grid(alpha=0.2)
        ax.set_xlim(args.tmin if args.tmin is not None else time[0],
                    args.tmax if args.tmax is not None else time[-1])

    part_label = "pure signal" if source is signals else "data (h + n)"
    axes[0, 0].set_title(f"{Path(args.injection).name} - {part_label}")
    axes[-1, 0].set_xlabel("Time relative to primary merger [s]")

    out_name = args.out or f"{Path(args.injection).stem}_time.png"
    return fig, out_name


def main():
    parser = argparse.ArgumentParser(
        description="Time-domain amplitude plot of arbitrary (possibly overlapping) "
                    "waveform signals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = parser.add_argument_group("source")
    src.add_argument("--dataset", default=DEFAULT_DATASET,
                     help="waveform dataset HDF5 to draw signals from")
    src.add_argument("--index", type=int, nargs="+", default=None,
                     help="sample indices to plot; more than one = overlapping signals")
    src.add_argument("--random", type=int, default=1,
                     help="number of random samples (used when --index is not given)")
    src.add_argument("--injection", default=None,
                     help="plot this injection HDF5 instead of dataset samples")
    src.add_argument("--part", choices=["signal", "data"], default="signal",
                     help="injection mode: pure signal (context/signal) or d = h + n")

    shape = parser.add_argument_group("signal placement")
    shape.add_argument("--shift", type=float, nargs="+", default=[],
                       help="per-signal merger offset in seconds (positive = later); "
                            "missing entries default to 0")
    shape.add_argument("--scale", type=float, nargs="+", default=[],
                       help="per-signal amplitude scale; missing entries default to 1")
    shape.add_argument("--polarization", choices=["h_cross", "h_plus"],
                       default="h_cross")
    shape.add_argument("--t_window", type=float, default=8.0,
                       help="length of the plotted segment [s] (dataset mode)")
    shape.add_argument("--end_margin", type=float, default=0.1,
                       help="seconds kept after the LAST merger (dataset mode adds "
                            "the largest positive --shift on top of this)")
    shape.add_argument("--seed", type=int, default=None,
                       help="seed for --random sample selection")

    look = parser.add_argument_group("appearance")
    look.add_argument("--whiten", action="store_true",
                      help="whiten by the ASD (dataset: ASD dataset; injection: "
                           "the ASD stored in the file)")
    look.add_argument("--observing_run", default="O3",
                      help="ASD dataset used for whitening in dataset mode")
    look.add_argument("--detector", nargs="+", default=["H1", "L1"],
                      help="injection: detectors to plot; dataset: first entry picks "
                           "the whitening ASD")
    look.add_argument("--no_components", action="store_true",
                      help="draw only the summed trace; by default overlapping signals "
                           "are shown as coloured lines on top of a grey 'sum' band, "
                           "so sum and single signals are equally visible")
    look.add_argument("--tmin", type=float, default=None, help="x-axis lower limit [s]")
    look.add_argument("--tmax", type=float, default=None, help="x-axis upper limit [s]")
    look.add_argument("--out", default=None, help="output file name")
    args = parser.parse_args()

    if args.injection:
        fig, out_name = plot_injection_signal(args)
    else:
        fig, out_name = plot_dataset_signal(args)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / out_name
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
