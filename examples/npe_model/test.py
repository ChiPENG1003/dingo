from construct_data import *
import numpy as np, matplotlib.pyplot as plt

data_type = "BNS_5"
sample_idx = 51
observing_run = "O2"
fmin, fmax = 10, 500

freq, h_freq, chirp_mass = get_signal_freq(data_type, sample_idx)
h_time, fs, f_start = freq_to_time_domain(h_freq, freq, chirp_mass)
print(f"chirp_mass={chirp_mass:.3f} Msun, f_start={f_start:.1f} Hz")

fq, tq, qm = q_transform_map(h_time, fs=fs, fmin=fmin, fmax=fmax,
                               q=12.7, n_freqs=128)
tq_rel = tq - tq[-1]   # 0 = merger

# 从 parameters HDF5 里读取 chirp mass（或手动指定）
import h5py
with h5py.File(f"training_data/waveform_dataset_{data_type}.hdf5", "r") as f:
    row = f["parameters"][sample_idx]
    params = {name: row[name] for name in f["parameters"].dtype.names}
print(params)

G_over_c3 = 4.93e-6   # G*M_sun/c^3 in seconds
Mc = chirp_mass
Mc_s = Mc * G_over_c3

t_chirp = np.linspace(-8.0, -0.05, 500)
f_chirp = (1/np.pi) * (5/(256 * np.abs(t_chirp)))**(3/8) / Mc_s**(5/8)

plt.figure(figsize=(10, 4))
plt.pcolormesh(tq_rel, fq, np.abs(qm), shading='gouraud', cmap='viridis')
plt.yscale('log'); plt.ylim(fmin, fmax)
plt.plot(t_chirp, f_chirp, 'r--', lw=1.5, label='theoretical chirp')
plt.legend(); plt.colorbar()
plt.title(f"BNS_5 sample {sample_idx}, Mc={Mc:.3f} M_sun")
plt.savefig("diag_chirp_track.png", dpi=150); plt.close()
print("saved")
