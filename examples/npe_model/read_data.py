import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

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

_WHITENING_ASD_PATH = "/home/users/p/peng4/.local/lib/python3.11/site-packages/bilby/gw/detector/noise_curves/aLIGO_ZERO_DET_high_P_asd.txt"

def get_signal_data(data_type="BNS", return_complex=False):
    data_path = f"training_data/waveform_dataset_{data_type}.hdf5"
    with h5py.File(data_path, "r") as f:
        h_cross_raw = f["polarizations/h_cross"][:]
        h_plus_raw  = f["polarizations/h_plus"][:]
        raw = f["parameters"][:]
        parameters = np.array(raw.tolist()) if raw.dtype.names else raw
        has_svd = "svd" in f and "V" in f["svd"]
        V = f["svd/V"][:] if has_svd else None          # (n_freq, size) SVD basis, complex

    if has_svd:
        # Decompress SVD coefficients back to full frequency-domain waveforms.
        h_cross = h_cross_raw @ V.conj().T  # (N_samples, n_freq)
        h_plus  = h_plus_raw  @ V.conj().T
    else:
        # SVD disabled: full-length whitened polarizations are stored directly.
        h_cross = h_cross_raw
        h_plus  = h_plus_raw

    delta_f = 0.125  # Hz
    freq = np.arange(h_cross.shape[1]) * delta_f

    ref_data = np.loadtxt(_WHITENING_ASD_PATH)
    asd_ref = np.interp(freq, ref_data[:, 0], ref_data[:, 1])
    h_cross = h_cross * asd_ref
    h_plus  = h_plus  * asd_ref

    if return_complex:
        return freq, h_cross, h_plus, parameters

    h_f = np.sqrt(np.abs(h_cross)**2 + np.abs(h_plus)**2)
    return freq, h_f, parameters

def get_signal_data_time(data_type="BNS"):
    data_path = f"training_data/waveform_dataset_{data_type}.hdf5"
    with h5py.File(data_path, "r") as f:
        h_cross_raw = f["polarizations/h_cross"][:]
        h_plus_raw  = f["polarizations/h_plus"][:]
        has_svd = "svd" in f and "V" in f["svd"]
        V = f["svd/V"][:] if has_svd else None           # (n_freq, size) SVD basis, complex

    if has_svd:
        h_cross_freq = h_cross_raw @ V.conj().T  # (N_samples, n_freq)
        h_plus_freq  = h_plus_raw  @ V.conj().T
    else:
        h_cross_freq = h_cross_raw
        h_plus_freq  = h_plus_raw

    delta_f = 0.125
    n_freq = h_cross_freq.shape[1]  # 8193
    N = 2 * (n_freq - 1)     # 16384
    delta_t = 1.0 / (N * delta_f)  # total duration = 1/delta_f = 8 s

    h_cross_t = np.fft.irfft(h_cross_freq, n=N, axis=-1)
    h_plus_t  = np.fft.irfft(h_plus_freq,  n=N, axis=-1)

    t = np.arange(N) * delta_t

    return t, h_cross_t, h_plus_t

def get_asd_data_time(observing_run="O1"):
    freq, mask, asd = get_asd_data(observing_run)
    delta_f = 0.125
    n_freq = len(freq)  # 16385
    N = 2 * (n_freq - 1)     # 32768
    delta_t = 1.0 / (N * delta_f)  # total duration = 1/delta_f = 8 s

    asd_time = np.fft.irfft(asd, n=N)

    t = np.arange(N) * delta_t

    return t, asd_time


def get_overlapping_signal(data_type_1="BNS", idx_1=0, data_type_2="BNS", idx_2=0, delta_t=0.0):
    freq, h_cross_1, h_plus_1, params_1 = get_signal_data(data_type_1, return_complex=True)
    freq, h_cross_2, h_plus_2, params_2 = get_signal_data(data_type_2, return_complex=True)

    time_phase = np.exp(-2j * np.pi * freq * delta_t)
    h_cross_total = h_cross_1[idx_1] + h_cross_2[idx_2] * time_phase
    h_plus_total  = h_plus_1[idx_1]  + h_plus_2[idx_2]  * time_phase

    h_f_total = np.sqrt(np.abs(h_cross_total)**2 + np.abs(h_plus_total)**2)
    return freq, h_f_total, params_1[idx_1], params_2[idx_2]


def calculate_m1m2(index, parameters):
    parameters = parameters[index]
    Mc = parameters[0]
    q = parameters[1]
    m1 = Mc * (1 + q)**0.2 / q**0.6
    m2 = m1 * q
    return m1, m2

init_param = 0
rand_param = np.random.randint(0, 100)


freq_1, h_f_1, params_1 = get_signal_data(data_type="BNS_1")
freq_2, h_f_2, params_2 = get_signal_data(data_type="BNS_2")
freq_3, h_f_3, params_3 = get_signal_data(data_type="BNS_3")
freq_4, h_f_4, params_4 = get_signal_data(data_type="BNS_4")
freq_5, h_f_5, params_5 = get_signal_data(data_type="BNS_5")
freq_6, h_f_6, params_6 = get_signal_data(data_type="BNS_6")
freq_7, h_f_7, params_7 = get_signal_data(data_type="BNS_7")
freq_8, h_f_8, params_8 = get_signal_data(data_type="BNS_8")
# Characteristic strain: plot 2*sqrt(f)*|h(f)| so the curves match the
# y-axis label and are directly comparable to the ASD sqrt(S_n(f)).
char = 2.0 * np.sqrt(freq_1)  # freq grids are identical across datasets

y_bns_1 = char * h_f_1[rand_param]
y_bns_2 = char * h_f_2[rand_param]
y_bns_3 = char * h_f_3[init_param]
y_bns_4 = char * h_f_4[rand_param]
y_bns_5 = char * h_f_5[rand_param]
y_bns_6 = char * h_f_6[rand_param]
y_bns_7 = char * h_f_7[init_param]
y_bns_8 = char * h_f_8[init_param]


plt.figure(figsize=(12, 8))
plt.title(f"ASD - Signal [{rand_param}]", fontsize=16)

mask_sig = (freq_2 > 1e1) & (freq_2 < 4096) & np.isfinite(y_bns_2) & (y_bns_2 > 0)
#plt.loglog(freq_1[mask_sig], y_bns_1[mask_sig], lw=0.6, label=f"BNS1{init_param}")
m21, m22 = calculate_m1m2(rand_param, params_2)
plt.loglog(freq_2[mask_sig], y_bns_2[mask_sig], lw=0.6, 
           label=f"(2) m=({m21:.2f}, {m22:.2f}), chirp_mass = {params_2[rand_param, 0]:.2f}, lambda=({params_2[rand_param, 3]:.2f}, {params_2[rand_param, 4]:.2f}), chi=({params_2[rand_param, 6]:.2f}, {params_2[rand_param, 7]:.2f})") # Good
m31, m32 = calculate_m1m2(rand_param, params_3)
plt.loglog(freq_3[mask_sig], y_bns_3[mask_sig], lw=0.6, 
           label=f"(3) m=({m31:.2f}, {m32:.2f}), chirp_mass = {params_3[init_param, 0]:.2f}, lambda=({params_3[init_param, 3]:.2f}, {params_3[init_param, 4]:.2f}), chi=({params_3[init_param, 6]:.2f}, {params_3[init_param, 7]:.2f})") # Good
m41, m42 = calculate_m1m2(rand_param, params_4)
plt.loglog(freq_4[mask_sig], y_bns_4[mask_sig], lw=0.6,
           label=f"(4) m=({m41:.2f}, {m42:.2f}), chirp_mass = {params_4[rand_param, 0]:.2f}, lambda=({params_4[rand_param, 3]:.2f}, {params_4[rand_param, 4]:.2f}), chi=({params_4[rand_param, 6]:.2f}, {params_4[rand_param, 7]:.2f})") # Good
m51, m52 = calculate_m1m2(rand_param, params_5)
plt.loglog(freq_5[mask_sig], y_bns_5[mask_sig], lw=0.6, 
           label=f"(5) m=({m51:.2f}, {m52:.2f}), chirp_mass = {params_5[rand_param, 0]:.2f}, lambda=({params_5[rand_param, 3]:.2f}, {params_5[rand_param, 4]:.2f}), chi=({params_5[rand_param, 6]:.2f}, {params_5[rand_param, 7]:.2f})")
m61, m62 = calculate_m1m2(rand_param, params_6)
plt.loglog(freq_6[mask_sig], y_bns_6[mask_sig], lw=0.6, 
           label=f"(6) m=({m61:.2f}, {m62:.2f}), chirp_mass = {params_6[rand_param, 0]:.2f}, lambda=({params_6[rand_param, 3]:.2f}, {params_6[rand_param, 4]:.2f}), chi=({params_6[rand_param, 6]:.2f}, {params_6[rand_param, 7]:.2f})") # Good
#m71, m72 = calculate_m1m2(rand_param, params_7)
#plt.loglog(freq_7[mask_sig], y_bns_7[mask_sig], lw=0.6, label=f"(GW170817ish)m1 = {m71:.2f}, m2 = {m72:.2f}")
#m81, m82 = calculate_m1m2(rand_param, params_8)
#plt.loglog(freq_8[mask_sig], y_bns_8[mask_sig], lw=0.6, label=f"m1 = {m81:.2f}, m2 = {m82:.2f}")

freq_asd, mask_asd, y_asd = get_asd_data(observing_run="O2")
plt.loglog(freq_asd[mask_asd], y_asd[mask_asd], lw=0.8, label="O2 ASD", color="black")

plt.xlim(10.25, 2048)
plt.ylim(1e-27, 1e-17)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$\sqrt{S_n(f)},\ 2\sqrt{f}\,|\tilde{h}(f)|$")
plt.legend()
plt.savefig("asd_comparison_and_signal.png", dpi=300)
plt.show()


choice = input("Do you want to see the combination of signals? (y/n): ")
if choice.lower() == 'y':
    rand_param2 = np.random.randint(0, 100)
    while rand_param2 == rand_param:
        rand_param2 = np.random.randint(0, 100)
    delta_t = float(input("Enter time offset delta_t for signal 2 (seconds, e.g. 0.5): "))

    freq_ov, h_f_ov, p1, p2 = get_overlapping_signal(
        data_type_1="BNS_2", idx_1=rand_param,
        data_type_2="BNS_4", idx_2=rand_param2,
        delta_t=delta_t,
    )
    m1_1, m2_1 = p1[0] * (1 + p1[1])**0.2 / p1[1]**0.6, p1[0] * (1 + p1[1])**0.2 / p1[1]**0.6 * p1[1]
    m1_2, m2_2 = p2[0] * (1 + p2[1])**0.2 / p2[1]**0.6, p2[0] * (1 + p2[1])**0.2 / p2[1]**0.6 * p2[1]
    delta_phi = p1[2] - p2[2]  # phase difference between the two signals (params index 2)

    mask_ov = (freq_ov > 1e1) & (freq_ov < 4096) & np.isfinite(h_f_ov) & (h_f_ov > 0)

    plt.figure(figsize=(12, 8))
    plt.title(f"Overlapping Signals [idx {rand_param} + {rand_param2}, $\Delta$t={delta_t:.2f}s]", fontsize=16)
    plt.loglog(freq_asd[mask_asd], y_asd[mask_asd], lw=0.8, label="O2 ASD", color="black")
    plt.loglog(freq_2[mask_sig], y_bns_2[mask_sig], lw=0.6, ls="--",
               label=f"(2)[{rand_param}] m=({m1_1:.2f},{m2_1:.2f}), $\phi$={p1[2]:.2f}")
    y_bns_4b = 2.0 * np.sqrt(freq_4) * h_f_4[rand_param2]
    mask_4 = (freq_4 > 1e1) & (freq_4 < 4096) & np.isfinite(y_bns_4b) & (y_bns_4b > 0)
    plt.loglog(freq_4[mask_4], y_bns_4b[mask_4], lw=0.6, ls="--",
               label=f"(4)[{rand_param2}] m=({m1_2:.2f},{m2_2:.2f}), $\phi$={p2[2]:.2f}")
    #plt.loglog(freq_ov[mask_ov], h_f_ov[mask_ov], lw=1.0, label=f"Overlap ($\Delta$$\phi$={delta_phi:.2f} rad, $\Delta$t={delta_t:.2f}s)")
    plt.xlim(10.25, 2048)
    plt.ylim(1e-27, 1e-17)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\sqrt{S_n(f)},\ 2\sqrt{f}\,|\tilde{h}(f)|$")
    plt.legend()
    plt.savefig("overlapping_signals.png", dpi=300)
    plt.show()
else:
    print("Combination of signals plot skipped.")