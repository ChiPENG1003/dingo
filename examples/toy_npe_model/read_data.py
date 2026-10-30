import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def get_asd_data(time_psd = 1024):
    asd_path = f"training_data/asd_dataset/asds_O1.hdf5"

    # Frequency settings from waveform_dataset_settings.yaml
    delta_f = 0.25
    asd_idx = 0

    with h5py.File(asd_path, "r") as f:
        asd_h1 = f["asds/H1"][asd_idx]
        asd_l1 = f["asds/L1"][asd_idx]
        print(f"ASD shapes: H1: {asd_h1.shape}, L1: {asd_l1.shape}")

    num_waveform_bins = asd_h1.shape[0]

    freq_full = np.arange(asd_h1.shape[0]) * delta_f
    freq = freq_full[:num_waveform_bins]
    asd_h1_seq = asd_h1[:num_waveform_bins]
    asd_l1_seq = asd_l1[:num_waveform_bins]

    # If ASD = sqrt(S_n), then S^{-1/2} = 1/ASD
    y = np.sqrt((1 / asd_h1_seq) ** (-2) + (1 / asd_l1_seq) ** (-2))
    mask = (freq > 1e1) & np.isfinite(y) & (y > 0)
    return freq, mask, y

def get_signal_data():
    data_path = "training_data/waveform_dataset.hdf5"
    with h5py.File(data_path, "r") as f:
        h_cross = f["polarizations/h_cross"][:]
        h_plus = f["polarizations/h_plus"][:]
        try:
            assert h_cross.shape[1] == h_plus.shape[1] == 4097
        except AssertionError:
            print("Error: Expected waveform length of 4097, but got", h_cross.shape[1], "and", h_plus.shape[1])
            raise

    freq_range = np.arange(h_cross.shape[1]) * 0.25
    freq = freq_range
    h_f = np.sqrt(np.abs(h_cross)**2 + np.abs(h_plus)**2)

    return freq, h_f

def calculte_snr(h_f, asd_freq, asd):
    # Interpolate ASD to match the frequency bins of the signal
    asd_interp = np.interp(h_f, asd_freq, asd)

    # Calculate SNR using the formula: SNR^2 = 4 * sum(|h(f)|^2 / S_n(f))
    snr_squared = 4 * np.sum((h_f / asd_interp) ** 2)
    snr = np.sqrt(snr_squared)

    return snr

def get_signal_data_time():
    data_path = "training_data/waveform_dataset.hdf5"
    with h5py.File(data_path, "r") as f:
        h_cross = f["polarizations/h_cross"][:]
        h_plus  = f["polarizations/h_plus"][:]

    delta_f = 0.25
    n_freq = 4097
    N = 2 * (n_freq - 1)
    delta_t = 1.0 / (N * delta_f)

    h_cross_t = np.fft.irfft(h_cross, n=N, axis=-1)
    h_plus_t  = np.fft.irfft(h_plus,  n=N, axis=-1)

    t = np.arange(N) * delta_t
    # t = (np.arange(N) - N) * delta_t  # -4 ~ 0 s

    return t, h_cross_t, h_plus_t


#freq_ref, mask_ref, y_ref = get_asd_data(time_psd = 1024)
freq, h_f = get_signal_data()
freq_2048, mask_2048, y_2048 = get_asd_data(time_psd = 2048)
#freq_4096, mask_4096, y_4096 = get_asd_data(time_psd = 4096)

fig, ax = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle("ASD Comparison for Different time_psd, and Signal in time domain", fontsize=16)
# Figure 1: ASD Comparison for different time_psd
#ax[0].loglog(freq_ref[mask_ref], y_ref[mask_ref], lw=0.8, label="1024s")
ax[0].loglog(freq_2048[mask_2048], y_2048[mask_2048], lw=0.8, label="2048s")
ax[0].loglog(freq[1:], h_f[0, 1:], lw=0.8, label="Signal (H1+L1)")
#ax[0].loglog(freq_4096[mask_4096], y_4096[mask_4096], lw=0.8, label="4096s")
ax[0].set_xlabel("Frequency [Hz]")
ax[0].set_xlim(20, 1024)
ax[0].set_ylabel(r"$S^{-1/2}\ [\mathrm{Hz}^{-1/2}]$")
ax[0].legend()

# Figure 2: Signal in time domain
t, h_cross_t, h_plus_t = get_signal_data_time()
#h = np.sqrt(h_cross_t**2 + h_plus_t**2)
ax[1].plot(t, h_cross_t[0], lw=0.8, label="H Cross")
ax[1].plot(t, h_plus_t[0], lw=0.8, label="H Plus")
#ax[1].plot(t, h[0], lw=0.8, label="Combined")
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel(r"$h(t)$")
ax[1].legend()

plt.savefig("asd_comparison_and_signal_time.png", dpi=300)
plt.show()
