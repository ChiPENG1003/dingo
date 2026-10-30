import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

import argparse


# This file lives in gnpe_model/construct_plot_data/. Anchor every project path
# to the gnpe_model directory rather than the working directory, so the script
# behaves the same however it is invoked.
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(HERE)
TRAINING_DATA_DIR = os.path.join(EXAMPLE_DIR, "training_data")
RESULT_ROOT = os.path.join(EXAMPLE_DIR, "inference", "inference_result")
# The previous default was "inference_result/injection.hdf5", which never
# existed; the file is under legacy/single/ after the directory reorganisation.
DEFAULT_INJECTION = os.path.join(RESULT_ROOT, "legacy", "single", "injection.hdf5")


DETECTOR_LABELS = {
    "H1": "LIGO-Hanford",
    "L1": "LIGO-Livingston",
}

F_MIN = 10.0                                    # Hz, lower edge of the analysis/whitening band
F_MAX = 1024.0                                  # Hz, upper edge (== domain.f_max)
DELTA_F = 0.125                                 # Hz (== domain.delta_f)
F_S = 2.0 * F_MAX                               # 2048 Hz time-domain sampling rate
NUM_FREQ_BINS = int(round(F_MAX / DELTA_F)) + 1  # 8193 bins: 0 ... F_MAX, matches SVD V

_WHITENING_ASD_PATH = Path(
    "/home/users/p/peng4/.local/lib/python3.11/site-packages/bilby/"
    "gw/detector/noise_curves/aLIGO_ZERO_DET_high_P_asd.txt"
)

def get_fixed_whitening_asd(freq):
    candidates = [_WHITENING_ASD_PATH]

    try:
        import bilby

        candidates.append(
            Path(bilby.__file__).parent
            / "gw/detector/noise_curves/aLIGO_ZERO_DET_high_P_asd.txt"
        )
    except ImportError:
        pass

    for path in candidates:
        if path.exists():
            ref_data = np.loadtxt(path)
            return np.interp(freq, ref_data[:, 0], ref_data[:, 1])

    raise FileNotFoundError(
        "Could not find aLIGO_ZERO_DET_high_P_asd.txt needed to undo "
        "waveform_dataset compression whitening."
    )

def get_asd_data(observing_run = "O1"):
    asd_path = os.path.join(TRAINING_DATA_DIR, "asd_dataset", f"asds_{observing_run}.hdf5")

    delta_f = DELTA_F
    num_waveform_bins = NUM_FREQ_BINS  # covers 0 ~ F_MAX Hz (= f_max / delta_f + 1)
    asd_idx = 0

    with h5py.File(asd_path, "r") as f:
        asd_h1 = f["asds/H1"][asd_idx]
        asd_l1 = f["asds/L1"][asd_idx]
        print(f"ASD shape: {asd_h1.shape}, {asd_l1.shape}")

    freq_full = np.arange(asd_h1.shape[0]) * delta_f
    freq = freq_full[:num_waveform_bins]
    asd_h1_seq = asd_h1[:num_waveform_bins]
    asd_l1_seq = asd_l1[:num_waveform_bins]

    y = np.sqrt(asd_h1_seq**2 + asd_l1_seq**2)
    mask = (freq >= F_MIN) & np.isfinite(y) & (y > 0)
    return freq, mask, y

def get_detector_asd_data(observing_run="O1", detector="H1"):
    asd_path = os.path.join(TRAINING_DATA_DIR, "asd_dataset", f"asds_{observing_run}.hdf5")

    delta_f = DELTA_F
    num_waveform_bins = NUM_FREQ_BINS  # 8193 bins, 0 ... F_MAX Hz
    asd_idx = 0

    with h5py.File(asd_path, "r") as f:
        available = set(f["asds"].keys())
        if detector not in available:
            raise KeyError(
                f"Detector {detector} not found in {asd_path}. "
                f"Available detectors: {sorted(available)}"
            )
        asd = f[f"asds/{detector}"][asd_idx]
        print(f"{detector} ASD shape: {asd.shape}")

    freq_full = np.arange(asd.shape[0]) * delta_f
    freq = freq_full[:num_waveform_bins]
    asd_seq = asd[:num_waveform_bins]

    mask = (freq >= F_MIN) & np.isfinite(asd_seq) & (asd_seq > 0)
    return freq, mask, asd_seq

def get_noise_time_from_asd(freq, mask, asd, seed=1234):

    n_freq = len(freq)
    n_time = 2 * (n_freq - 1)
    delta_f = freq[1] - freq[0]
    fs = n_time * delta_f
    delta_t = 1.0 / fs

    asd = np.asarray(asd, dtype=float)
    bad = (~np.isfinite(asd)) | (asd <= 0)
    if np.any(bad):
        asd[bad] = np.nanmedian(asd[mask])

    rng = np.random.default_rng(seed)
    re = rng.normal(0.0, 1.0, n_freq)
    im = rng.normal(0.0, 1.0, n_freq)

    noise_freq = (re + 1j * im) * asd * np.sqrt(fs * n_time / 4.0)
    noise_freq[0] = rng.normal(0.0, asd[0] * np.sqrt(fs * n_time / 2.0))
    noise_freq[-1] = rng.normal(0.0, asd[-1] * np.sqrt(fs * n_time / 2.0))

    asd_time = np.fft.irfft(noise_freq, n=n_time)
    time = np.arange(n_time) * delta_t

    return time, asd_time, fs

def get_asd_time(observing_run="O1", seed=1234):
    freq, mask, asd = get_asd_data(observing_run)
    return get_noise_time_from_asd(freq, mask, asd, seed=seed)


def get_signal_freq(sample_idx=0, polarization="h_cross"):
    data_path = os.path.join(TRAINING_DATA_DIR, "waveform_dataset_500K.hdf5")

    with h5py.File(data_path, "r") as f:
        raw = f[f"polarizations/{polarization}"][sample_idx : sample_idx + 1]
        has_svd = "svd" in f and "V" in f["svd"]
        V = f["svd/V"][:] if has_svd else None
        chirp_mass = float(f["parameters"][sample_idx]["chirp_mass"])

    # SVD-compressed datasets store coefficients; otherwise full polarizations.
    h_freq = raw @ V.conj().T if has_svd else raw

    delta_f = DELTA_F
    freq = np.arange(h_freq.shape[1]) * delta_f
    h_freq = h_freq * get_fixed_whitening_asd(freq)

    return freq, h_freq[0], chirp_mass


def freq_to_time_domain(h_freq, freq, chirp_mass, t_window=8.0, t_end_margin=0.1,
                        t_extend=None):
    G_over_c3 = 4.93e-6  # G * M_sun / c^3  [seconds]
    Mc_s = chirp_mass * G_over_c3

    T_long = t_extend if (t_extend is not None and t_extend > t_window) else t_window

    # f_start: frequency at which 0PN chirp is T_long seconds before merger
    f_start = (1.0 / np.pi) * (5.0 / (256.0 * T_long)) ** (3.0 / 8.0) / Mc_s ** (5.0 / 8.0)

    delta_f_orig = freq[1] - freq[0]
    f_max = freq[-1]
    fs = 2.0 * f_max  # sampling rate stays fixed at 4096 Hz

    if T_long > t_window:
        # Interpolate h_freq to a finer frequency grid (delta_f_new = 1/T_long)
        # so the IRFFT covers T_long seconds without changing fs.
        delta_f_new = 1.0 / T_long
        freq_new = np.arange(0, f_max + delta_f_new, delta_f_new)
        amp = np.abs(h_freq)
        phase = np.unwrap(np.angle(h_freq))
        amp_new = np.interp(freq_new, freq, amp)
        phase_new = np.interp(freq_new, freq, phase)
        h_work = amp_new * np.exp(1j * phase_new)
        n_time_long = int(round(T_long * fs))
        if n_time_long % 2:
            n_time_long += 1
    else:
        freq_new = freq
        h_work = h_freq.copy()
        n_time_long = 2 * (len(freq) - 1)

    # zero below f_start with smooth Hann taper
    h_work[freq_new < f_start] = 0.0
    f_taper_end = 1.5 * f_start
    mask_taper = (freq_new >= f_start) & (freq_new < f_taper_end)
    if np.any(mask_taper):
        t_vals = (freq_new[mask_taper] - f_start) / (f_taper_end - f_start)
        h_work[mask_taper] *= 0.5 * (1.0 - np.cos(np.pi * t_vals))

    # time-shift: merger at T_long - t_end_margin from the start of the array
    t_shift = T_long - t_end_margin
    h_work *= np.exp(-2j * np.pi * freq_new * t_shift)

    h_time_long = fs * np.fft.irfft(h_work, n=n_time_long)

    # extract the last t_window seconds (the segment containing the merger)
    n_extract = int(round(t_window * fs))
    h_time = h_time_long[-n_extract:]

    return h_time, fs, f_start

def get_signal_time(sample_idx=0, polarization="h_cross"):
    freq, h_freq, chirp_mass = get_signal_freq(sample_idx, polarization)
    h_time, fs, _ = freq_to_time_domain(h_freq, freq, chirp_mass)
    time = np.arange(len(h_time)) / fs
    return time, h_time, fs

def frequency_domain_asd_whiten(x_time, asd, fs, f_min=F_MIN, f_max=F_MAX, eps=1e-60):
    x_time = np.asarray(x_time, dtype=np.float64)
    asd = np.asarray(asd, dtype=np.float64)

    n_time = len(x_time)
    x_freq = np.fft.rfft(x_time, n=n_time)
    freq = np.fft.rfftfreq(n_time, d=1.0 / fs)
    n_freq = min(len(x_freq), len(asd))

    white_freq = np.zeros_like(x_freq)
    white_freq[:n_freq] = x_freq[:n_freq] / (asd[:n_freq] + eps)

    out_of_band = (freq < f_min) | (freq > f_max)
    white_freq[out_of_band[: len(white_freq)]] = 0.0

    return np.fft.irfft(white_freq, n=n_time)

def compute_optimal_snr(h_freq, asd, delta_f, fmin=None, fmax=None, eps=1e-60):
    h_freq = np.asarray(h_freq)
    asd = np.asarray(asd, dtype=np.float64)

    n = min(len(h_freq), len(asd))
    h_freq = h_freq[:n]
    asd = asd[:n]
    freq = np.arange(n) * delta_f

    mask = np.isfinite(asd) & (asd > 0)
    if fmin is not None:
        mask &= freq >= fmin
    if fmax is not None:
        mask &= freq <= fmax

    psd = asd[mask] ** 2
    snr_squared = 4.0 * np.sum(np.abs(h_freq[mask]) ** 2 / (psd + eps)) * delta_f
    return np.sqrt(max(snr_squared, 0.0))

def q_transform_map(
    x_time,
    fs,
    fmin=F_MIN,
    fmax=F_MAX,
    q=12.7,
    n_freqs=256,
    hop_fraction=0.25,
    n_time_bins=1000,
):
    x_time = np.asarray(x_time)
    n = len(x_time)
    freqs = np.geomspace(fmin, fmax, n_freqs)

    rows = []
    time_grids = []

    for f0 in freqs:
        nperseg = int(round(q / f0 * fs))
        nperseg = max(nperseg, 16)

        if nperseg >= n:
            nperseg = n - 1
        if nperseg % 2 == 1:
            nperseg += 1

        hop = max(1, int(round(nperseg * hop_fraction)))
        starts = np.arange(0, n - nperseg, hop)

        if len(starts) < 2:
            rows.append(np.full(1, np.nan + 0j))
            time_grids.append(np.full(1, np.nan))
            continue

        window = np.hanning(nperseg)
        t_local = np.arange(nperseg) / fs
        basis = window * np.exp(-2j * np.pi * f0 * t_local)
        norm = np.sqrt(np.sum(window**2))

        row = np.asarray(
            [
                np.sum(x_time[s : s + nperseg] * basis) / norm
                for s in starts
            ]
        )
        rows.append(row)
        time_grids.append((starts + nperseg / 2) / fs)

    t_common = np.linspace(0, n / fs, n_time_bins)
    qmap = np.zeros((len(freqs), len(t_common)), dtype=np.complex128)

    for i, row in enumerate(rows):
        t_row = time_grids[i]
        valid = np.isfinite(row.real) & np.isfinite(row.imag) & np.isfinite(t_row)
        if np.sum(valid) < 2:
            qmap[i, :] = np.nan + 0j
        else:
            real = np.interp(
                t_common,
                t_row[valid],
                row[valid].real,
                left=row[valid][0].real,
                right=row[valid][-1].real,
            )
            imag = np.interp(
                t_common,
                t_row[valid],
                row[valid].imag,
                left=row[valid][0].imag,
                right=row[valid][-1].imag,
            )
            qmap[i, :] = real + 1j * imag

    return freqs, t_common, qmap

def normalize_amplitude_frequencywise(amp, eps=1e-30):
    background = np.nanmedian(amp, axis=1, keepdims=True)
    bad = (~np.isfinite(background)) | (background <= eps)
    background[bad] = eps
    return amp / background

def print_scale_comparison(label, noise, signal, eps=1e-300):
    noise_std = np.std(noise)
    signal_std = np.std(signal)
    noise_rms = np.sqrt(np.mean(noise**2))
    signal_rms = np.sqrt(np.mean(signal**2))
    noise_max = np.max(np.abs(noise))
    signal_max = np.max(np.abs(signal))

    print(f"[{label}]")
    print(f"  ASD/noise std   = {noise_std:.3e}")
    print(f"  signal std      = {signal_std:.3e}")
    print(f"  std ratio       = {noise_std / (signal_std + eps):.3e}")
    print(f"  ASD/noise rms   = {noise_rms:.3e}")
    print(f"  signal rms      = {signal_rms:.3e}")
    print(f"  rms ratio       = {noise_rms / (signal_rms + eps):.3e}")
    print(f"  ASD/noise max   = {noise_max:.3e}")
    print(f"  signal max      = {signal_max:.3e}")
    print(f"  max ratio       = {noise_max / (signal_max + eps):.3e}")

def plot_time_frequency(
    sample_idx=0,
    polarization="h_cross",
    observing_run="O2",
    signal_scale=1.0,
    seed=1234,
    fmin=F_MIN,
    fmax=F_MAX,
    q=12.7,
    n_freqs=256,
):
    freq, h_freq, chirp_mass = get_signal_freq(sample_idx, polarization)
    h_time, fs, _ = freq_to_time_domain(h_freq, freq, chirp_mass)

    time_asd, asd_time, fs_asd = get_asd_time(observing_run, seed=seed)
    _, _, asd = get_asd_data(observing_run)

    n = min(len(h_time), len(asd_time))
    h_time = h_time[:n]
    asd_time = asd_time[:n]

    white_asd = frequency_domain_asd_whiten(asd_time, asd, fs_asd)
    unscaled_white_signal = frequency_domain_asd_whiten(h_time, asd, fs)
    total_signal_scale = signal_scale

    print(f"signal scale applied = {total_signal_scale:.3e}")

    scaled_h_time = total_signal_scale * h_time
    asd_with_signal = asd_time + scaled_h_time

    white_signal = total_signal_scale * unscaled_white_signal
    white_asd_with_signal = frequency_domain_asd_whiten(asd_with_signal, asd, fs_asd)
    signal_freq = np.fft.rfft(scaled_h_time, n=n) / fs_asd
    delta_f = fs_asd / n
    optimal_snr = compute_optimal_snr(
        signal_freq,
        asd,
        delta_f,
        fmin=fmin,
        fmax=fmax,
    )

    print("=" * 30)
    print(f"optimal SNR = {optimal_snr:.3f}")
    print(f"q value = {q}")
    print("=" * 30)

    print_scale_comparison("raw time series", asd_time, scaled_h_time)
    print_scale_comparison("ASD-normalized time series", white_asd, white_signal)

    freq, t, qmap = q_transform_map(
        white_signal,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        q=q,
        n_freqs=n_freqs,
    )
    freq_asd, t_asd, qmap_asd = q_transform_map(
        white_asd_with_signal,
        fs=fs_asd,
        fmin=fmin,
        fmax=fmax,
        q=q,
        n_freqs=n_freqs,
    )
    t = t - t[-1]
    t_asd = t_asd - t_asd[-1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=False, sharey=True)


    #amp = normalize_amplitude_frequencywise(np.abs(qmap))
    amp = np.abs(qmap)
    im0 = axes[0].pcolormesh(
        t,
        freq,
        amp,
        shading="auto",
        cmap="viridis",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylim(fmin, fmax)

    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].set_title(f"Q-map [{sample_idx}](SNR = {optimal_snr:.1f})")
    fig.colorbar(im0, ax=axes[0], label="Normalized Amplitude")


    #amp_asd = normalize_amplitude_frequencywise(np.abs(qmap_asd))
    amp_asd = np.abs(qmap_asd)
    im1 = axes[1].pcolormesh(
        t_asd,
        freq_asd,
        amp_asd,
        shading="auto",
        cmap="viridis",
    )
    axes[1].set_yscale("log")
    axes[1].set_ylim(fmin, fmax)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_title(f"{observing_run} whitened ASD")
    fig.colorbar(im1, ax=axes[1], label="Normalized Amplitude")

    plt.tight_layout()

    out_dir = Path("picture/time_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tf_[{sample_idx}].png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")

def plot_detector_qmap_figure(
    sample_idx=0,
    polarization="h_cross",
    observing_run="O2",
    detectors=("H1", "L1"),
    signal_scale=1.0,
    seed=1234,
    fmin=20,
    fmax=1000,
    q=12.7,
    n_freqs=256,
    time_window=None,
    vmax=None,
    overlap_second=False,
    sample_idx_2=0,
    signal_scale_2=1.0,
    delta_t=0.0,
):
    _, h_time, fs = get_signal_time(sample_idx, polarization)

    if overlap_second:
        freq2, h_freq2, _ = get_signal_freq(sample_idx_2, polarization)
        h_freq2_shifted = h_freq2 * np.exp(-2j * np.pi * freq2 * delta_t)
        n_time2 = 2 * (len(freq2) - 1)
        fs2 = n_time2 * (freq2[1] - freq2[0])
        if fs2 != fs:
            raise ValueError(f"Sampling rates do not match: {fs} vs {fs2}")
        h_time2 = fs2 * np.fft.irfft(h_freq2_shifted, n=n_time2)
        n = min(len(h_time), len(h_time2))
        h_time = h_time[:n] * signal_scale + h_time2[:n] * signal_scale_2
        signal_scale = 1.0

    qmaps = []

    for i, detector in enumerate(detectors):
        freq_asd, mask_asd, asd = get_detector_asd_data(
            observing_run,
            detector=detector,
        )
        _, noise_time, fs_noise = get_noise_time_from_asd(
            freq_asd,
            mask_asd,
            asd,
            seed=seed + i,
        )

        n = min(len(h_time), len(noise_time))
        h_det = h_time[:n]
        noise_time = noise_time[:n]

        white_noise = frequency_domain_asd_whiten(noise_time, asd, fs_noise)
        unscaled_white_signal = frequency_domain_asd_whiten(h_det, asd, fs)
        total_signal_scale = signal_scale

        scaled_signal = total_signal_scale * h_det
        white_signal = total_signal_scale * unscaled_white_signal
        white_data = frequency_domain_asd_whiten(
            noise_time + scaled_signal,
            asd,
            fs_noise,
        )
        signal_freq = np.fft.rfft(scaled_signal, n=n) / fs_noise
        delta_f = fs_noise / n
        optimal_snr = compute_optimal_snr(
            signal_freq,
            asd,
            delta_f,
            fmin=fmin,
            fmax=fmax,
        )

        print(f"[{detector}] signal scale applied = {total_signal_scale:.3e}")
        print(f"[{detector}] optimal SNR = {optimal_snr:.3f}")
        print_scale_comparison(f"{detector} raw time series", noise_time, scaled_signal)
        print_scale_comparison(
            f"{detector} ASD-normalized time series",
            white_noise,
            white_signal,
        )

        freqs, t, qmap = q_transform_map(
            white_data,
            fs=fs_noise,
            fmin=fmin,
            fmax=fmax,
            q=q,
            n_freqs=n_freqs,
        )
        amp = np.abs(qmap)
        qmaps.append((detector, freqs, t - t[-1], amp))

    fig, axes = plt.subplots(
        len(qmaps),
        1,
        figsize=(10.0, 8.0 if len(qmaps) > 1 else 4.6),
        sharex=True,
        sharey=True,
        constrained_layout=False,
    )
    if len(qmaps) == 1:
        axes = [axes]

    im = None
    for ax, (detector, freqs, t_rel, amp) in zip(axes, qmaps):
        color_limits = {}
        if vmax is not None:
            color_limits = {"vmin": 0, "vmax": vmax}
        im = ax.pcolormesh(
            t_rel,
            freqs,
            amp,
            shading="gouraud",
            cmap="viridis",
            **color_limits,
        )
        ax.set_yscale("log")
        ax.set_ylim(fmin, fmax)
        ax.set_yticks([50, 100, 500])
        ax.set_yticklabels(["50", "100", "500"])
        ax.text(
            0.05,
            0.82,
            DETECTOR_LABELS.get(detector, detector),
            transform=ax.transAxes,
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )

        if time_window is not None:
            ax.set_xlim(-time_window, 0)
        else:
            ax.set_xlim(t_rel[0], 0)

    axes[-1].set_xlabel("Time (seconds)", fontsize=14)
    fig.text(0.02, 0.5, "Frequency (Hz)", va="center", rotation="vertical", fontsize=14)

    single = len(qmaps) == 1
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10 if not single else 0.16,
                        top=0.88 if not single else 0.80, hspace=0.10)
    cax = fig.add_axes([0.16, 0.90 if not single else 0.86, 0.81, 0.025])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(f"Normalized amplitude (SNR = {optimal_snr:.1f})", fontsize=14)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_ticks([0, 2, 4, 6])

    out_dir = Path("picture/time_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{detectors[0]}" if len(qmaps) == 1 else ""
    out_path = out_dir / f"detector_qmap[{sample_idx}]{suffix}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


def read_injection(injection_path=None):
    """Return (data, asds, signals) from an injection HDF5.

    data[det], asds[det], signals[det] are frequency-domain arrays.
    data holds d = h + n (signal + noise); signals holds the pure signal h,
    written as context/signal by inference.py. signals is empty ({}) for older
    injection files that predate the pure-signal storage, and None for real
    events (construct_real_event.py), where no pure signal exists at all.
    """
    injection_path = injection_path or DEFAULT_INJECTION
    with h5py.File(injection_path, "r") as f:
        data = {det: f[f"context/waveform/{det}"][:] for det in f["context/waveform"]}
        asds = {det: f[f"context/asds/{det}"][:] for det in f["context/asds"]}
        if "context/signal" in f:
            signals = {det: f[f"context/signal/{det}"][:] for det in f["context/signal"]}
        elif "event_metadata/real_event" in f:
            signals = None
        else:
            signals = {}
    return data, asds, signals


def read_event_metadata(injection_path=None):
    """The event_metadata group as a plain dict ({} if the file has none)."""
    injection_path = injection_path or DEFAULT_INJECTION
    meta = {}
    with h5py.File(injection_path, "r") as f:
        if "event_metadata" in f:
            for name, dset in f["event_metadata"].items():
                value = dset[()]
                meta[name] = value.decode() if isinstance(value, bytes) else value
    return meta


# Template bank for the real-event matched filter. Chirp mass spans the BBH range;
# two mass ratios are enough because the SNR is a shallow function of q. Validated
# against the published network SNRs: GW170729 10.19 vs 10.2, GW170809 12.34 vs
# 12.4, GW150914 23.28 vs 24.4.
_MF_CHIRP_MASSES = np.geomspace(6.0, 70.0, 15)
_MF_MASS_RATIOS = (0.4, 1.0)


def _matched_filter_snr(x_freq, asd, fmin=F_MIN, fmax=F_MAX):
    """Matched-filter SNR of real data, maximised over template mass and time.

    A real event has no injected waveform, so `compute_optimal_snr` cannot be
    used. Matched filtering maximised over amplitude, phase and time needs only
    the template *shape*, i.e. the masses -- not the distance, inclination or
    coalescence phase -- so a coarse mass bank is enough to recover the published
    SNR to a few percent.

    Two caveats, both physical rather than numerical:
      * A loud glitch is matched better than the signal. GW170817 L1 returns ~364
        because the filter locks onto the ETMY saturation glitch; that event needs
        gating first.
      * The bank stops at the BBH range. A BNS inspiral from 20 Hz lasts ~150 s
        and cannot fit the 8 s segment, so it is out of scope here.

    Returns None if pycbc is unavailable.
    """
    try:
        from pycbc.filter import matched_filter
        from pycbc.types import FrequencySeries
        from pycbc.waveform import get_fd_waveform
    except ImportError:
        return None

    data = FrequencySeries(np.asarray(x_freq, dtype=np.complex128), delta_f=DELTA_F)
    psd = FrequencySeries(np.asarray(asd, dtype=np.float64) ** 2, delta_f=DELTA_F)

    best = 0.0
    for mc in _MF_CHIRP_MASSES:
        for q in _MF_MASS_RATIOS:
            eta = q / (1 + q) ** 2
            m_total = mc / eta**0.6
            hp, _ = get_fd_waveform(
                approximant="IMRPhenomD",
                mass1=m_total / (1 + q),
                mass2=m_total * q / (1 + q),
                delta_f=DELTA_F,
                f_lower=fmin,
                f_final=fmax,
                distance=1000.0,
            )
            hp.resize(len(data))
            rho = float(
                abs(
                    matched_filter(
                        hp, data, psd=psd,
                        low_frequency_cutoff=fmin, high_frequency_cutoff=fmax,
                    )
                ).max()
            )
            best = max(best, rho)
    return best


def _injection_detector_qmap(
    data, asds, det, signals=None, fmin=F_MIN, fmax=F_MAX, q=12.7, n_freqs=256,
    end_margin_s=0.1, merger_idx=None,
):
    asd = asds[det]
    x_freq = data[det]
    if signals is None:
        # Real event: no template, so no optimal SNR can be defined here.
        snr_freq = None
    elif signals and det in signals:
        snr_freq = signals[det]
    else:
        snr_freq = x_freq
        print(
            f"[WARN] {det}: no pure signal (context/signal) in injection file; "
            "SNR computed from d = h + n is noise-dominated (~sqrt(2 N_bins)) and "
            "NOT a real SNR. Regenerate the injection to store the pure signal."
        )

    n_time = 2 * (len(x_freq) - 1)
    fs = n_time * DELTA_F
    x_time = fs * np.fft.irfft(x_freq, n=n_time)
    white = frequency_domain_asd_whiten(x_time, asd, fs, f_min=fmin, f_max=fmax)

    if merger_idx is None:
        # Anchor on the PRIMARY (unshifted) merger: the strong sample whose wrapped
        # buffer time is closest to 0, not the global-loudest (which can be the louder
        # signal 2).
        env = np.abs(white)
        tau = np.arange(n_time) / fs
        tau = np.where(tau >= (n_time / fs) / 2.0, tau - n_time / fs, tau)
        strong = np.where(env > 0.25 * env.max())[0]
        merger_idx = int(strong[np.argmin(np.abs(tau[strong]))])
    target_idx = n_time - 1 - int(round(end_margin_s * fs))
    white = np.roll(white, target_idx - merger_idx)

    freqs, t, qmap = q_transform_map(
        white, fs=fs, fmin=fmin, fmax=fmax, q=q, n_freqs=n_freqs
    )
    amp = np.abs(qmap)
    if snr_freq is None:
        snr, snr_kind = _matched_filter_snr(x_freq, asd, fmin, fmax), "matched-filter SNR"
    else:
        snr = compute_optimal_snr(snr_freq, asd, DELTA_F, fmin=fmin, fmax=fmax)
        snr_kind = "SNR"
    return freqs, t - target_idx / fs, amp, snr, snr_kind


def plot_injection_qmap(
    injection_path=None,
    detectors=("H1", "L1"),
    fmin=F_MIN,
    fmax=F_MAX,
    q=12.7,
    n_freqs=256,
    time_window=None,
    vmax=None,
    end_margin_s=None,
):
    """Per-detector time-frequency (Q-transform) map of the injected data d = h + n."""
    injection_path = injection_path or DEFAULT_INJECTION
    data, asds, signals = read_injection(injection_path)
    meta = read_event_metadata(injection_path)

    if meta.get("real_event"):
        # Real data: the trigger sits at buffer index 0 by construction (dingo shifts
        # it there), so anchor on that rather than hunting for the loudest sample.
        # Keeping time_buffer seconds after the trigger reproduces the segment exactly
        # as downloaded, which puts the Tukey taper -- an unavoidable ~0.8 s hole where
        # the two segment ends meet -- back at the edges of the axis instead of
        # 6 s before the merger.
        merger_idx = 0
        if end_margin_s is None:
            end_margin_s = float(meta.get("time_buffer", 2.0))
    else:
        merger_idx = None  # injections are cyclic: no segment edge to worry about
        if end_margin_s is None:
            end_margin_s = 0.1

    if detectors is None:
        detectors = tuple(data.keys())

    qmaps = []
    for det in detectors:
        freqs, t_rel, amp, snr, snr_kind = _injection_detector_qmap(
            data, asds, det, signals=signals, fmin=fmin, fmax=fmax, q=q,
            n_freqs=n_freqs, end_margin_s=end_margin_s, merger_idx=merger_idx,
        )
        print(f"[{det}] {snr_kind} = {snr:.2f}" if snr is not None else f"[{det}] no SNR")
        qmaps.append((det, freqs, t_rel, amp, snr, snr_kind))

    fig, axes = plt.subplots(
        len(qmaps), 1, figsize=(10.0, 8.0 if len(qmaps) > 1 else 4.6),
        sharex=True, sharey=True,
    )
    if len(qmaps) == 1:
        axes = [axes]

    im = None
    snr_square = 0.0
    for ax, (det, freqs, t_rel, amp, snr, snr_kind) in zip(axes, qmaps):
        if snr is not None:
            snr_square += snr**2
        color_limits = {"vmin": 0, "vmax": vmax} if vmax is not None else {}
        im = ax.pcolormesh(
            t_rel, freqs, amp, shading="gouraud", cmap="viridis", **color_limits,
        )
        ax.set_yscale("log")
        ax.set_ylim(fmin, fmax)
        # ax.set_yticks([50, 100, 500])
        # ax.set_yticklabels(["50", "100", "500"])
        ax.text(
            0.05, 0.82,
            f"{DETECTOR_LABELS.get(det, det)}"
            + (f"\n({snr_kind} = {snr:.1f})" if snr is not None else ""),
            transform=ax.transAxes, fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
        )
        if time_window is not None:
            ax.set_xlim(-time_window, t_rel[-1])
        else:
            ax.set_xlim(t_rel[0], t_rel[-1])

    axes[-1].set_xlabel("Time from merger (seconds)", fontsize=14)
    fig.text(0.02, 0.5, "Frequency (Hz)", va="center", rotation="vertical", fontsize=14)

    # A one-panel figure is shorter, so the colourbar strip and its top label need
    # proportionally more room than in the stacked two-detector layout. Everything
    # is pulled down a little from the earlier layout to leave a band for the title.
    single = len(qmaps) == 1
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10 if not single else 0.16,
                        top=0.85 if not single else 0.76, hspace=0.10)
    cax = fig.add_axes([0.16, 0.87 if not single else 0.82, 0.81, 0.025])

    # Network SNR: independent detectors, so the per-detector values add in
    # quadrature (same convention as the published GWTC network SNRs).
    net_snr = np.sqrt(snr_square) if snr_square > 0 else None
    fig.suptitle(
        "Signal qmap" + (f" (SNR = {net_snr:.1f})" if net_snr is not None else ""),
        fontsize=15, y=0.985,
    )

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label("Injection data (signal + noise)  whitened amplitude", fontsize=13)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")

    out_dir = Path("picture/time_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{detectors[0]}" if len(qmaps) == 1 else ""
    out_path = out_dir / f"{Path(injection_path).stem}_qmap{suffix}.png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")
    if net_snr is not None:
        print(f"Value of SNR: {net_snr:.3f}")


def plot_overlap_comparison(
    injection_paths,
    detectors=("H1", "L1"),
    fmin=F_MIN,
    fmax=F_MAX,
    q=12.7,
    n_freqs=256,
    time_window=None,
    vmax=None,
    out_name="overlap_comparison.png",
):
    """Grid of injection Q-maps for several overlap shifts in ONE figure.

    injection_paths: list of (label, path). Rows = injections (shifts), columns =
    detectors. A shared colour scale (common vmax) makes panels directly comparable.
    """
    # Compute every panel first so a common vmax can be derived.
    grid = []  # grid[r][c] = (freqs, t_rel, amp, snr, snr_kind, det)
    for label, path in injection_paths:
        data, asds, signals = read_injection(path)
        dets = detectors if detectors is not None else tuple(data.keys())
        row = [
            _injection_detector_qmap(
                data, asds, det, signals=signals, fmin=fmin, fmax=fmax, q=q, n_freqs=n_freqs
            )
            + (det,)
            for det in dets
        ]
        grid.append((label, row))

    if vmax is None:
        all_amp = np.concatenate([panel[2].ravel() for _, row in grid for panel in row])
        vmax = float(np.nanpercentile(all_amp, 99.5))

    nrows = len(grid)
    ncols = len(grid[0][1])
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 2.7 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )

    im = None
    for r, (label, row) in enumerate(grid):
        for c, (freqs, t_rel, amp, snr, snr_kind, det) in enumerate(row):
            ax = axes[r][c]
            im = ax.pcolormesh(
                t_rel, freqs, amp, shading="gouraud", cmap="viridis",
                vmin=0, vmax=vmax,
            )
            ax.set_yscale("log")
            ax.set_ylim(fmin, fmax)
            ax.text(
                0.05, 0.80,
                f"{DETECTOR_LABELS.get(det, det)}\n{label}"
                + (f"\n({snr_kind} = {snr:.1f})" if snr is not None else ""),
                transform=ax.transAxes, fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "0.75", "alpha": 0.9},
            )
            if time_window is not None:
                ax.set_xlim(-time_window, t_rel[-1])
            else:
                ax.set_xlim(t_rel[0], t_rel[-1])

    for c in range(ncols):
        axes[-1][c].set_xlabel("Time from merger (seconds)", fontsize=13)
    fig.text(0.02, 0.5, "Frequency (Hz)", va="center", rotation="vertical", fontsize=13)

    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.09, top=0.90, hspace=0.12, wspace=0.06)
    cax = fig.add_axes([0.10, 0.93, 0.87, 0.02])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(f"Injection data (signal + noise) [whitened amplitude]", fontsize=12)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")

    out_dir = Path("picture/time_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


# Overlap results have been named three ways over time. Newest first; the older
# two only matter for files produced before inference.py was reorganised.
_SHIFT_PATTERNS = (
    re.compile(r"_shift(\d+(?:\.\d+)?)s"),               # ..._d0080mpc_shift0.050s
    re.compile(r"_(\d+(?:\.\d+)?)s_"),                   # injection_overlap_0.05s_80mpc
    re.compile(r"^injection_overlap_(\d+(?:\.\d+)?)$"),  # injection_overlap_0.1
)


def _shift_from_name(stem):
    """Seconds encoded in a result filename, or None if it is not an overlap result."""
    for pattern in _SHIFT_PATTERNS:
        m = pattern.search(stem)
        if m:
            return float(m.group(1))
    return None


def _discover_overlap_injections(result_dir=None):
    """Return [(label, path), ...] for overlap injections, sorted by shift.

    Searches the current layout written by inference.py --
    inference_result/posteriors/<model>/overlap/ -- and also the two directories
    earlier versions wrote to, so results produced before the reorganisation stay
    usable. The model name goes into the label because a single sweep may now be
    present for more than one network.
    """
    root = Path(result_dir or RESULT_ROOT)
    search = [
        root / "posteriors",     # current: inference.py output, keyed by model
        root / "legacy",         # results from before the directory reorganisation
    ]

    found = {}
    for base in search:
        if not base.is_dir():
            continue
        for p in base.rglob("*.hdf5"):
            shift = _shift_from_name(p.stem)
            if shift is None:
                continue
            # posteriors/<model>/overlap/<file> -> label it with <model>
            parts = p.relative_to(base).parts
            model = parts[0] if base.name == "posteriors" and len(parts) > 1 else None
            label = f"shift {shift:g} s" + (f" [{model}]" if model else "")
            found.setdefault((model, shift), (label, str(p)))

    return [found[k] for k in sorted(found, key=lambda k: (k[0] or "", k[1]))]


def plot_all_bns(
    sample_idx=0,
    polarization="h_cross",
    observing_run="O3",
    signal_scale=1.0,
    seed=1234,
):
    plot_time_frequency(
        sample_idx=sample_idx,
        polarization=polarization,
        observing_run=observing_run,
        signal_scale=signal_scale,
        seed=seed,
    )


if __name__ == "__main__":
    INJECTION_PATH = os.path.join(RESULT_ROOT, "posteriors", "ET", "single", "seed0053_dprior.hdf5")
    random_phase_shift = np.random.uniform(-4, 0)

    parser = argparse.ArgumentParser()
    #parser.add_argument("--sample_idx", type=int, default = 40, help="Index of the sample to plot")
    parser.add_argument("--polarization", type=str, default="h_cross", help="Polarization of the signal")
    parser.add_argument("--observing_run", type=str, default="ET", help="Observing run")
    parser.add_argument("--signal_scale", type=float, default=1.0, help="Scale of the signal")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--phase_shift", type=float, default=random_phase_shift,
                        help="Phase shift for the signal")
    parser.add_argument("--plot_type", choices=["tf", "dqmap", "inj", "inj_compare"],
                        help="tf: time-frequency plot; dqmap: detector q-map plot; "
                             "inj: injection time-frequency q-map; "
                             "inj_compare: grid of overlap shifts in one figure")
    parser.add_argument("--injection_path", type=str,
                        default=INJECTION_PATH,
                        help="Injection HDF5 (for --plot_type inj)")
    parser.add_argument("--injection_paths", type=str, nargs="+", default=None,
                        help="inj_compare: overlap HDF5 files to tile in one figure "
                             "(default: auto-discover inference_result/injection_overlap_*.hdf5)")
    parser.add_argument("--detector", type=str, nargs="+", default=["H1", "L1"],
                        help="detectors to plot; pass a single one (e.g. --detector H1) "
                             "for a one-panel figure")
    parser.add_argument("--time_window", type=float, default=None,
                        help="inj: seconds before merger to show (zoom); default full 8 s")
    parser.add_argument("--end_margin", type=float, default=None,
                        help="inj: seconds of data kept after the merger. Defaults to "
                             "0.1 for injections and to the event's time_buffer for "
                             "real data. The last ~0.3 s of the buffer is "
                             "flat-extrapolated by the Q-transform, so raise this "
                             "(e.g. 1.0) to see the merger itself resolved.")
    args = parser.parse_args()

    if args.plot_type == "tf":
        plot_all_bns(
            sample_idx=np.random.randint(0,49999),
            #sample_idx=args.sample_idx,
            polarization=args.polarization,
            observing_run=args.observing_run,
            signal_scale=args.signal_scale,
            seed=args.seed,
        )

    elif args.plot_type == "dqmap":
        plot_detector_qmap_figure(
            sample_idx=51,
            polarization=args.polarization,
            observing_run=args.observing_run,
            signal_scale=args.signal_scale,
            seed=args.seed,
            detectors=tuple(args.detector),
            fmin=20,
            fmax=1000,
            overlap_second=True,
            sample_idx_2=np.random.randint(0, 49999),
            signal_scale_2=args.signal_scale,
            delta_t=args.phase_shift,
        )

    elif args.plot_type == "inj":
        plot_injection_qmap(
            injection_path=args.injection_path,
            detectors=tuple(args.detector),
            time_window=args.time_window,
            end_margin_s=args.end_margin,
        )

    elif args.plot_type == "inj_compare":
        if args.injection_paths:
            injection_paths = []
            for p in args.injection_paths:
                shift = _shift_from_name(Path(p).stem)
                injection_paths.append(
                    (f"shift {shift:g} s" if shift is not None else Path(p).stem, p)
                )
        else:
            injection_paths = _discover_overlap_injections()
        if not injection_paths:
            raise SystemExit("No overlap injections found to compare.")
        plot_overlap_comparison(
            injection_paths,
            detectors=tuple(args.detector),
            time_window=args.time_window,
        )
