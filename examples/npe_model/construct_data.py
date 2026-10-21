import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List

import argparse



DETECTOR_LABELS = {
    "H1": "LIGO-Hanford",
    "L1": "LIGO-Livingston",
}

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
    asd_path = f"training_data/asd_dataset/asds_{observing_run}.hdf5"

    delta_f = 0.125
    num_waveform_bins = 16385  # covers 0 ~ 2048 Hz (= f_max / delta_f + 1, f_s=8192)
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
    mask = (freq > 1e1) & np.isfinite(y) & (y > 0)
    return freq, mask, y

def get_detector_asd_data(observing_run="O1", detector="H1"):
    asd_path = f"training_data/asd_dataset/asds_{observing_run}.hdf5"

    delta_f = 0.125
    num_waveform_bins = 16385
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

    mask = (freq > 1e1) & np.isfinite(asd_seq) & (asd_seq > 0)
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


def get_signal_freq(data_type="BNS_1", sample_idx=0, polarization="h_cross"):
    data_path = f"training_data/waveform_dataset_{data_type}.hdf5"

    with h5py.File(data_path, "r") as f:
        raw = f[f"polarizations/{polarization}"][sample_idx : sample_idx + 1]
        has_svd = "svd" in f and "V" in f["svd"]
        V = f["svd/V"][:] if has_svd else None
        chirp_mass = float(f["parameters"][sample_idx]["chirp_mass"])

    # SVD-compressed datasets store coefficients; otherwise full polarizations.
    h_freq = raw @ V.conj().T if has_svd else raw

    delta_f = 0.125
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

def get_signal_time(data_type="BNS_1", sample_idx=0, polarization="h_cross"):
    freq, h_freq, chirp_mass = get_signal_freq(data_type, sample_idx, polarization)
    h_time, fs, _ = freq_to_time_domain(h_freq, freq, chirp_mass)
    time = np.arange(len(h_time)) / fs
    return time, h_time, fs

def frequency_domain_asd_whiten(x_time, asd, fs, eps=1e-60):
    x_time = np.asarray(x_time, dtype=np.float64)
    asd = np.asarray(asd, dtype=np.float64)

    n_time = len(x_time)
    x_freq = np.fft.rfft(x_time, n=n_time)
    n_freq = min(len(x_freq), len(asd))

    white_freq = np.zeros_like(x_freq)
    white_freq[:n_freq] = x_freq[:n_freq] / (asd[:n_freq] + eps)

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
    fmin=10,
    fmax=2048,
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
    data_type="BNS_1",
    sample_idx=0,
    polarization="h_cross",
    observing_run="O2",
    signal_scale=1.0,
    seed=1234,
    fmin=10,
    fmax=2048,
    q=12.7,
    n_freqs=256,
    phase_shift=1.0,
):
    data_types = list(data_type) if isinstance(data_type, (list, tuple)) else [data_type]
    sample_indices = (
        list(sample_idx) if isinstance(sample_idx, (list, tuple)) else [sample_idx] * len(data_types)
    )
    if len(sample_indices) != len(data_types):
        raise ValueError("sample_idx must be a scalar or have the same length as data_type.")

    signal_parts = []
    fs_values = []
    for i, (data_type_i, sample_idx_i) in enumerate(zip(data_types, sample_indices)):
        freq_i, h_freq_i, chirp_mass_i = get_signal_freq(data_type_i, sample_idx_i, polarization)

        if len(data_types) == 2 and i == 1:
            h_freq_i = h_freq_i * np.exp(-2j * np.pi * freq_i * phase_shift)

        h_time_i, fs_i, _ = freq_to_time_domain(h_freq_i, freq_i, chirp_mass_i)

        signal_parts.append(h_time_i)
        fs_values.append(fs_i)

    if len(set(fs_values)) != 1:
        raise ValueError(f"All signals must have the same sampling rate, got {fs_values}.")

    fs = fs_values[0]
    n_signal = min(len(h_time_i) for h_time_i in signal_parts)
    h_time = np.sum([h_time_i[:n_signal] for h_time_i in signal_parts], axis=0)
    data_type_label = "+".join(data_types)
    sample_idx_label = "+".join(str(i) for i in sample_indices)
    
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

    per_signal_snrs = []
    if len(signal_parts) > 1:
        for h_part in signal_parts:
            h_part_freq = np.fft.rfft(total_signal_scale * h_part[:n], n=n) / fs_asd
            per_signal_snrs.append(
                compute_optimal_snr(h_part_freq, asd, delta_f, fmin=fmin, fmax=fmax)
            )
        snr_abs = 0.0
        for label, snr_i in zip(data_types, per_signal_snrs):
            print(f"{label} SNR = {snr_i:.3f}")
            snr_abs += snr_i**2

        snr_net = np.sqrt(snr_abs)
        print(f"  {label} SNR = {snr_net:.3f}")

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
    if per_signal_snrs:
        snr_str = " / ".join(
            f"{dt}({snr:.1f})" for dt, snr in zip(data_types, per_signal_snrs)
        )
        snr_net = np.sqrt(sum(snr ** 2 for snr in per_signal_snrs))
        snr_str = f"{snr_str}, net={snr_net:.1f}"
    else:
        snr_str = f"{optimal_snr:.1f}"
    axes[0].set_title(f"Q-map [{sample_idx_label}](SNR = {snr_str})")
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
    out_path = out_dir / (
        f"tf_{data_type_label}[{sample_idx_label}].png"
    )
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")

def plot_detector_qmap_figure(
    data_type="BNS_1",
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
    data_type_2=None,
    sample_idx_2=0,
    signal_scale_2=1.0,
    delta_t=0.0,
):
    _, h_time, fs = get_signal_time(data_type, sample_idx, polarization)

    if data_type_2 is not None:
        freq2, h_freq2 = get_signal_freq(data_type_2, sample_idx_2, polarization)
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
        figsize=(10.0, 8.0),
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

    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.10, top=0.88, hspace=0.10)
    cax = fig.add_axes([0.16, 0.90, 0.81, 0.025])
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cbar.set_label(f"Normalized amplitude (SNR = {optimal_snr:.1f})", fontsize=14)
    cbar.ax.xaxis.set_label_position("top")
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.set_ticks([0, 2, 4, 6])

    out_dir = Path("picture/time_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"detector_qmap[{sample_idx}].png"
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")

def plot_all_bns(
    sample_idx=0,
    polarization="h_cross",
    observing_run="O2",
    signal_scale=1.0,
    seed=1234,
    phase_shift=1.0,
    labels: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
):
    for i in labels:
        if isinstance(i, (list, tuple)):
            data_type = [f"BNS_{j}" for j in i]
        else:
            data_type = f"BNS_{i}"

        plot_time_frequency(
            data_type=data_type,
            sample_idx=sample_idx,
            polarization=polarization,
            observing_run=observing_run,
            signal_scale=signal_scale,
            seed=seed,
            phase_shift=phase_shift,
        )


if __name__ == "__main__":

    random_phase_shift = np.random.uniform(-4, 0)

    parser = argparse.ArgumentParser()
    #parser.add_argument("--sample_idx", type=int, default = 40, help="Index of the sample to plot")
    parser.add_argument("--polarization", type=str, default="h_cross", help="Polarization of the signal")
    parser.add_argument("--observing_run", type=str, default="O2", help="Observing run")
    parser.add_argument("--signal_scale", type=float, default=1.0, help="Scale of the signal")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--phase_shift", type=float, default=random_phase_shift, 
                        help="Phase shift for the signal")
    parser.add_argument("--plot_type", choices=["tf", "dqmap"], 
                        help="tf: time-frequency plot; dqmap: detector q-map plot")
    args = parser.parse_args()

    if args.plot_type == "tf":
        plot_all_bns(
            #sample_idx=np.random.randint(0,100),
            #sample_idx=args.sample_idx,
            sample_idx=85,
            polarization=args.polarization,
            observing_run=args.observing_run,
            signal_scale=args.signal_scale,
            seed=args.seed,
            phase_shift=args.phase_shift,
            labels=[3,5],
        )

    elif args.plot_type == "dqmap":
        plot_detector_qmap_figure(
            data_type="BNS_5",
            sample_idx=51,
            polarization=args.polarization,
            observing_run=args.observing_run,
            signal_scale=args.signal_scale,
            seed=args.seed,
            detectors=("H1", "L1"),
            fmin=20,
            fmax=1000,
            data_type_2="BNS_2",
            sample_idx_2=np.random.randint(0, 100),
            signal_scale_2=args.signal_scale,
            delta_t=args.phase_shift,
        )
