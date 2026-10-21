import h5py
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import List


# ============================================================
# PSD DATA
# ============================================================

def get_psd_data(observing_run="O1"):
    asd_path = f"training_data/asd_dataset/asds_{observing_run}.hdf5"

    delta_f = 0.125
    num_waveform_bins = 16385
    asd_idx = 0

    with h5py.File(asd_path, "r") as f:
        asd_h1 = f["asds/H1"][asd_idx]
        asd_l1 = f["asds/L1"][asd_idx]
        print(f"ASD shape: {asd_h1.shape}, {asd_l1.shape}")

    freq_full = np.arange(asd_h1.shape[0]) * delta_f
    freq = freq_full[:num_waveform_bins]

    asd_h1_seq = asd_h1[:num_waveform_bins]
    asd_l1_seq = asd_l1[:num_waveform_bins]

    psd = asd_h1_seq**2 + asd_l1_seq**2

    mask = (freq > 1e1) & np.isfinite(psd) & (psd > 0)

    return freq, mask, psd


# ============================================================
# COLORED GAUSSIAN NOISE FROM PSD
# ============================================================

def get_psd_time(observing_run="O1", seed=1234):
    freq, mask, psd = get_psd_data(observing_run)

    rng = np.random.default_rng(seed)

    n_freq = len(freq)
    n_time = 2 * (n_freq - 1)

    delta_f = freq[1] - freq[0]
    fs = n_time * delta_f
    delta_t = 1.0 / fs

    psd = np.asarray(psd, dtype=float)

    bad = (~np.isfinite(psd)) | (psd <= 0)
    if np.any(bad):
        psd[bad] = np.nanmedian(psd[mask])

    re = rng.normal(0.0, 1.0, n_freq)
    im = rng.normal(0.0, 1.0, n_freq)

    noise_freq = (re + 1j * im) * np.sqrt(psd * fs * n_time / 4.0)

    noise_freq[0] = rng.normal(
        0.0,
        np.sqrt(psd[0] * fs * n_time / 2.0),
    )

    noise_freq[-1] = rng.normal(
        0.0,
        np.sqrt(psd[-1] * fs * n_time / 2.0),
    )

    noise_time = np.fft.irfft(noise_freq, n=n_time)

    time = np.arange(n_time) * delta_t

    return time, noise_time, fs


# ============================================================
# SIGNAL
# ============================================================

def get_signal_time(data_type="BNS_1", sample_idx=0, polarization="h_cross"):
    data_path = f"training_data/waveform_dataset_{data_type}.hdf5"

    with h5py.File(data_path, "r") as f:
        coeffs = f[f"polarizations/{polarization}"][sample_idx : sample_idx + 1]
        V = f["svd/V"][:]

    h_freq = coeffs @ V.conj().T
    h_freq = h_freq[0]

    delta_f = 0.125
    freq = np.arange(len(h_freq)) * delta_f

    n_freq = len(freq)
    n_time = 2 * (n_freq - 1)

    fs = n_time * delta_f
    delta_t = 1.0 / fs

    h_time = np.fft.irfft(h_freq, n=n_time)

    time = np.arange(n_time) * delta_t

    return time, h_time, fs, h_freq


# ============================================================
# WHITENING
# ============================================================

def whiten_frequency_series(h_freq, psd, fs, eps=1e-60):
    n = min(len(h_freq), len(psd))

    h_freq = h_freq[:n]
    psd = psd[:n]

    white_freq = h_freq / (np.sqrt(psd * fs / 2.0) + eps)

    n_time = 2 * (n - 1)

    white_time = np.fft.irfft(white_freq, n=n_time)

    return white_time


# ============================================================
# Q-TRANSFORM-LIKE MAP
# ============================================================

def q_transform_like_map(
    x_time,
    fs,
    fmin=20,
    fmax=1024,
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
        window_duration = q / f0
        nperseg = int(round(window_duration * fs))

        nperseg = max(nperseg, 16)

        if nperseg >= n:
            nperseg = n - 1

        if nperseg % 2 == 1:
            nperseg += 1

        hop = max(1, int(round(nperseg * hop_fraction)))

        starts = np.arange(0, n - nperseg, hop)

        if len(starts) < 2:
            row = np.full(1, np.nan)
            t_row = np.full(1, np.nan)
        else:
            window = np.hanning(nperseg)

            t_local = np.arange(nperseg) / fs

            basis = window * np.exp(-2j * np.pi * f0 * t_local)

            coeffs = []

            norm = np.sqrt(np.sum(window**2))

            for s in starts:
                segment = x_time[s : s + nperseg]
                coeff = np.sum(segment * basis) / norm
                coeffs.append(np.abs(coeff))

            row = np.asarray(coeffs)

            t_row = (starts + nperseg / 2) / fs

        rows.append(row)
        time_grids.append(t_row)

    t_common = np.linspace(0, n / fs, n_time_bins)

    amp = np.zeros((len(freqs), len(t_common)))

    for i, row in enumerate(rows):
        t_row = time_grids[i]

        valid = np.isfinite(row) & np.isfinite(t_row)

        if np.sum(valid) < 2:
            amp[i, :] = np.nan
        else:
            amp[i, :] = np.interp(
                t_common,
                t_row[valid],
                row[valid],
                left=np.nan,
                right=np.nan,
            )

    return freqs, t_common, amp


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_amplitude_frequencywise(amp, eps=1e-30):
    background = np.nanmedian(amp, axis=1, keepdims=True)

    bad = (~np.isfinite(background)) | (background <= eps)

    background[bad] = eps

    return amp / background


def normalize_energy_frequencywise(amp, eps=1e-30):
    energy = amp**2

    mean_energy = np.nanmean(energy, axis=1, keepdims=True)

    bad = (~np.isfinite(mean_energy)) | (mean_energy <= eps)

    mean_energy[bad] = eps

    return energy / mean_energy


# ============================================================
# PLOT
# ============================================================

def plot_time_frequency(
    data_type="BNS_1",
    sample_idx=0,
    polarization="h_cross",
    observing_run="O2",
    signal_time_shift=0.0,
    fmin=20,
    fmax=500,
    seed=1234,
    q=12.7,
    n_freqs=256,
    plot_quantity="amplitude",  # "amplitude" or "energy"
):
    time, h_time, fs, h_freq = get_signal_time(
        data_type,
        sample_idx,
        polarization,
    )

    freq_psd, _, psd = get_psd_data(observing_run)

    time_noise, noise_time, fs_noise = get_psd_time(
        observing_run,
        seed=seed,
    )

    n_freq = min(len(h_freq), len(psd))

    n_time = 2 * (n_freq - 1)

    h_freq = h_freq[:n_freq]
    psd = psd[:n_freq]

    noise_time = noise_time[:n_time]

    noise_freq = np.fft.rfft(
        noise_time,
        n=n_time,
    )[:n_freq]

    data_freq = noise_freq + h_freq

    white_signal_time = whiten_frequency_series(
        h_freq,
        psd,
        fs,
    )

    white_data_time = whiten_frequency_series(
        data_freq,
        psd,
        fs_noise,
    )

    white_noise_time = whiten_frequency_series(
        noise_freq,
        psd,
        fs_noise,
    )

    print(f"signal RMS       = {np.std(h_time):.3e}")
    print(f"noise RMS        = {np.std(noise_time):.3e} physical")
    print(f"white noise RMS  = {np.std(white_noise_time):.3e}")
    print(f"signal time shift= {signal_time_shift:.3e} s")

    freqs_sig, t_sig, amp_sig = q_transform_like_map(
        white_signal_time,
        fs=fs,
        fmin=fmin,
        fmax=fmax,
        q=q,
        n_freqs=n_freqs,
    )

    freqs_data, t_data, amp_data = q_transform_like_map(
        white_data_time,
        fs=fs_noise,
        fmin=fmin,
        fmax=fmax,
        q=q,
        n_freqs=n_freqs,
    )

    if plot_quantity == "energy":
        z_sig = normalize_energy_frequencywise(amp_sig)
        z_data = normalize_energy_frequencywise(amp_data)

        color_label = "Normalized energy"
        vmin = 0
        vmax = 16

    else:
        z_sig = normalize_amplitude_frequencywise(amp_sig)
        z_data = normalize_amplitude_frequencywise(amp_data)

        color_label = "Normalized amplitude"
        vmin = 0
        vmax = 6

    duration = n_time / fs

    t_sig = t_sig - duration + signal_time_shift
    t_data = t_data - duration + signal_time_shift

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 7),
        sharex=True,
        sharey=True,
    )

    im0 = axes[0].pcolormesh(
        t_sig,
        freqs_sig,
        z_sig,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_yscale("log")
    axes[0].set_ylim(fmin, fmax)
    axes[0].set_ylabel("Frequency [Hz]")
    axes[0].set_title(
        f"{data_type} {polarization} whitened Q-map [{sample_idx}]"
    )

    fig.colorbar(
        im0,
        ax=axes[0],
        label=color_label,
    )

    im1 = axes[1].pcolormesh(
        t_data,
        freqs_data,
        z_data,
        shading="auto",
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_yscale("log")
    axes[1].set_ylim(fmin, fmax)
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")
    axes[1].set_title(
        f"{observing_run} whitened noise + {data_type} Q-map"
    )

    axes[1].set_xlim(
        -duration + signal_time_shift,
        0.5 + signal_time_shift,
    )

    fig.colorbar(
        im1,
        ax=axes[1],
        label=color_label,
    )

    plt.tight_layout()

    out_dir = Path("picture/time_frequency")
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / (
        f"qmap_{plot_quantity}_{data_type}_{polarization}_{sample_idx}_{observing_run}.png"
    )

    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"Saved {out_path}")


# ============================================================
# BATCH
# ============================================================

def plot_all_bns(
    sample_idx=0,
    polarization="h_cross",
    observing_run="O2",
    signal_time_shift=0.0,
    labels: List[int] = [1, 2, 3, 4, 5, 6, 7, 8],
    plot_quantity="amplitude",
):
    for i in labels:
        plot_time_frequency(
            data_type=f"BNS_{i}",
            sample_idx=sample_idx,
            polarization=polarization,
            observing_run=observing_run,
            signal_time_shift=signal_time_shift,
            plot_quantity=plot_quantity,
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    sample_idx = 34

    polarization = "h_cross"

    observing_run = "O2"

    signal_time_shift = 0.0

    plot_quantity = "amplitude"
    # plot_quantity = "energy"

    plot_all_bns(
        sample_idx=sample_idx,
        polarization=polarization,
        observing_run=observing_run,
        signal_time_shift=signal_time_shift,
        labels=[2],
        plot_quantity=plot_quantity,
    )