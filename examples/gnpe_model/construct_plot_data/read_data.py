import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random
import os

# This file lives in gnpe_model/construct_plot_data/; anchor project paths to
# the gnpe_model directory instead of the working directory.
HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE_DIR = os.path.dirname(HERE)
TRAINING_DATA_DIR = os.path.join(EXAMPLE_DIR, "training_data")
RESULT_ROOT = os.path.join(EXAMPLE_DIR, "inference", "inference_result")
DEFAULT_INJECTION = os.path.join(RESULT_ROOT, "legacy", "single", "injection.hdf5")

DELTA_F = 0.125
NUM_WAVEFORM_BINS = 16385  # covers 0 ~ 2048 Hz (= f_max / delta_f + 1, f_s=8192)


def get_detector_asds(observing_run="O3", detectors=("H1", "L1"),
                      asd_dir="asd_dataset", asd_idx=0):
    """
    Read one ASD per detector from an ASD dataset.

    Parameters
    ----------
    observing_run : str
        Run label used in the file name, e.g. 'O2', 'O3' or 'ET'.
    detectors : sequence of str
        Detector keys inside the file, e.g. ('H1', 'L1') or ('ET1', 'ET2', 'ET3').
    asd_dir : str
        Sub-directory of training_data holding the dataset. The ET design-curve
        dataset lives in 'asd_dataset_ET'.
    asd_idx : int
        Which ASD of the dataset to take. The ET dataset holds a single
        deterministic design curve, so only index 0 exists there.

    Returns
    -------
    freq : np.ndarray
    asds : dict of str -> np.ndarray
    """
    asd_path = os.path.join(TRAINING_DATA_DIR, asd_dir, f"asds_{observing_run}.hdf5")
    with h5py.File(asd_path, "r") as f:
        asds = {det: f[f"asds/{det}"][asd_idx] for det in detectors}
    n = min(NUM_WAVEFORM_BINS, min(a.shape[0] for a in asds.values()))
    print(f"{observing_run} ASD shapes: " +
          ", ".join(f"{d} {a.shape}" for d, a in asds.items()))
    return np.arange(n) * DELTA_F, {d: a[:n] for d, a in asds.items()}


def combine_asds(asds, method="network"):
    """
    Merge per-detector ASDs into a single curve.

    method='network' (the physical one)
        1/S_eff = sum_I 1/S_I,  i.e.  ASD_eff = (sum_I ASD_I**-2)**-0.5
        This is the combination whose single-curve SNR reproduces the network
        SNR, SNR_net**2 = sum_I SNR_I**2. For N detectors with the same ASD it
        gives ASD/sqrt(N): the three ET interferometers are sqrt(3) = 1.73x more
        sensitive than one of them.

    method='null' (the quadrature sum, kept for backwards compatibility)
        sqrt(sum_I ASD_I**2). This is the noise of the *summed* data streams, not
        a sensitivity. For ET it is precisely the null-stream noise: the three
        detector tensors sum to zero, so ET1+ET2+ET3 cancels the GW signal and
        keeps 3x the noise power. It is sqrt(3) times *worse* than a single
        interferometer, so it must not be read as a network sensitivity curve.

    Note that neither form includes the antenna patterns, which are
    direction-dependent; both are the equal-response (or sky-averaged) idealisation.
    """
    stack = np.vstack(list(asds.values()) if isinstance(asds, dict) else asds)
    if method == "network":
        return np.sum(stack**-2.0, axis=0) ** -0.5
    elif method == "null":
        return np.sqrt(np.sum(stack**2.0, axis=0))
    raise ValueError(f"unknown method {method!r}")


def get_asd_data(observing_run="O3", detectors=("H1", "L1"), asd_dir="asd_dataset",
                 method="null"):
    # method defaults to 'null' so that existing figures are unchanged; see
    # combine_asds -- 'network' is the one that means "sensitivity of the network".
    freq, asds = get_detector_asds(observing_run, detectors, asd_dir)
    y = combine_asds(asds, method=method)
    mask = (freq > 1e1) & np.isfinite(y) & (y > 0)
    return freq, mask, y

_BILBY_NOISE_CURVES = "/home/users/p/peng4/.local/lib/python3.11/site-packages/bilby/gw/detector/noise_curves"
_WHITENING_ASD_PATH = os.path.join(_BILBY_NOISE_CURVES, "aLIGO_ZERO_DET_high_P_asd.txt")
# ET waveform datasets are whitened with the ET-D curve (see
# waveform_dataset_settings_ET.yaml), so un-whitening them needs that same file.
_ET_WHITENING_ASD_PATH = os.path.join(EXAMPLE_DIR, "ET_D_asd.txt")


def get_signal_data(return_complex=False, n_max=200, data_path=None,
                    whitening_asd=None):
    # Only reconstruct the first `n_max` samples: the full dataset would need
    # tens of GB after the SVD matmul, and we only plot one sample below.
    #
    # `whitening_asd` must be the curve the dataset was compressed with
    # (compression.whitening in the waveform dataset settings), otherwise the
    # reconstructed strain is off by the ratio of the two curves.
    data_path = data_path or os.path.join(TRAINING_DATA_DIR, "waveform_dataset_1M.hdf5")
    whitening_asd = whitening_asd or _WHITENING_ASD_PATH
    with h5py.File(data_path, "r") as f:
        h_cross_coeffs = f["polarizations/h_cross"][:n_max]
        h_plus_coeffs  = f["polarizations/h_plus"][:n_max]
        raw = f["parameters"][:n_max]
        parameters = np.array(raw.tolist()) if raw.dtype.names else raw
        V = f["svd/V"][:]

    h_cross = h_cross_coeffs @ V.conj().T  # (N_samples, 8193)
    h_plus  = h_plus_coeffs  @ V.conj().T

    delta_f = 0.125  # Hz
    freq = np.arange(h_cross.shape[1]) * delta_f

    ref_data = np.loadtxt(whitening_asd)
    asd_ref = np.interp(freq, ref_data[:, 0], ref_data[:, 1])
    h_cross = h_cross * asd_ref
    h_plus  = h_plus  * asd_ref

    if return_complex:
        return freq, h_cross, h_plus, parameters

    h_f = np.sqrt(np.abs(h_cross)**2 + np.abs(h_plus)**2)
    return freq, h_f, parameters


def calculate_m1m2(index, parameters):
    parameters = parameters[index]
    Mc = parameters[0]
    q = parameters[1]
    m1 = Mc * (1 + q)**0.2 / q**0.6
    m2 = m1 * q
    return m1, m2


def get_injection_data(injection_path=None):
    injection_path = injection_path or DEFAULT_INJECTION
    with h5py.File(injection_path, "r") as f:
        h1 = f["context/waveform/H1"][:]
        l1 = f["context/waveform/L1"][:]
        asd_h1 = f["context/asds/H1"][:]
        asd_l1 = f["context/asds/L1"][:]
        params = {k: f["injection_parameters"][k][()]
                  for k in f["injection_parameters"]}

    delta_f = 0.125  # Hz, matches domain.delta_f in the settings
    freq = np.arange(h1.shape[0]) * delta_f

    h_f = np.sqrt(np.abs(h1) ** 2 + np.abs(l1) ** 2)
    asd = np.sqrt(asd_h1 ** 2 + asd_l1 ** 2)
    return freq, h_f, asd, params

init_param = 0
rand_param = np.random.randint(0, 100)


# BBH analysis band (matches waveform_dataset_settings.yaml: f_min=20, f_max=1024).
F_MIN = 20.0
F_MAX = 1024.0

freq_2, h_f_2, params_2 = get_signal_data()
y_sig_2 = h_f_2[rand_param]


plt.figure(figsize=(12, 8))
plt.title(f"ASD - Signal [{rand_param}]", fontsize=16)

# mask_sig = (freq_2 >= F_MIN) & (freq_2 <= F_MAX) & np.isfinite(y_sig_2) & (y_sig_2 > 0)
# m21, m22 = calculate_m1m2(rand_param, params_2)
# plt.loglog(freq_2[mask_sig], y_sig_2[mask_sig], lw=0.6,
#            label=f"m=({m21:.2f}, {m22:.2f}), chirp_mass={params_2[rand_param, 0]:.2f}, "
#                  f"chi=({params_2[rand_param, 3]:.2f}, {params_2[rand_param, 4]:.2f}), "
#                  f"tilt=({params_2[rand_param, 5]:.2f}, {params_2[rand_param, 6]:.2f})")

# freq_asd, mask_asd, y_asd = get_asd_data(observing_run="O3")
# plt.loglog(freq_asd[mask_asd], y_asd[mask_asd], lw=0.8, label="O3 ASD", color="black")

freq_inj, h_f_inj, asd_inj, inj_params = get_injection_data()
mask_inj = (freq_inj >= F_MIN) & (freq_inj <= F_MAX) & np.isfinite(h_f_inj) & (h_f_inj > 0)
Mc_inj, q_inj = inj_params["chirp_mass"], inj_params["mass_ratio"]
m1_inj = Mc_inj * (1 + q_inj) ** 0.2 / q_inj ** 0.6
m2_inj = m1_inj * q_inj
plt.loglog(freq_inj[mask_inj], h_f_inj[mask_inj], lw=0.8, color="crimson",
           label=f"injection m=({m1_inj:.2f}, {m2_inj:.2f}), "
                 f"chirp_mass={Mc_inj:.2f}, "
                 f"chi=({inj_params['a_1']:.2f}, {inj_params['a_2']:.2f}), "
                 f"d_L={inj_params['luminosity_distance']:.1f} Mpc")

mask_inj_asd = (freq_inj >= F_MIN) & (freq_inj <= F_MAX) & np.isfinite(asd_inj) & (asd_inj > 0)
plt.loglog(freq_inj[mask_inj_asd], asd_inj[mask_inj_asd], lw=0.8, color="gray",
           ls="--", label="injection ASD")

plt.xlim(F_MIN, F_MAX)
plt.ylim(1e-27, 1e-17)
plt.xlabel("Frequency [Hz]")
plt.ylabel(r"$\sqrt{S_n(f)},\ |\tilde{h}(f)|$")
plt.legend()
plt.savefig("asd_comparison_and_signal.png", dpi=300)
plt.show()


# ---------------------------------------------------------------------------
# ET: design ASD of the three interferometers, against O2 and a signal.
# ---------------------------------------------------------------------------

def plot_et_asd_and_signal(signal_index=None, out_name="picture/ET_asd_and_signal.png"):
    """
    Plot the ET-D ASD of ET1/ET2/ET3 together with a BBH signal, with the O2
    H1/L1 ASDs shown for scale.

    The strain is taken from the aLIGO-whitened 1M waveform dataset and
    un-whitened back to physical units, so it does not depend on which detector
    network it will be projected onto. Once training_data/waveform_dataset_ET.hdf5
    exists, pass data_path/whitening_asd below to read it instead.
    """
    if signal_index is None:
        signal_index = rand_param

    freq_et, asds_et = get_detector_asds("ET", ("ET1", "ET2", "ET3"), "asd_dataset_ET")
    freq_o2, asds_o2 = get_detector_asds("O2", ("H1", "L1"), "asd_dataset")

    freq_sig, h_f_sig, params_sig = get_signal_data()
    y_sig = h_f_sig[signal_index]
    m1, m2 = calculate_m1m2(signal_index, params_sig)
    d_ref = 1000.0

    band = lambda f: (f >= 5.0) & (f <= 2048.0)
    mask_sig = (freq_sig >= F_MIN) & (freq_sig <= F_MAX) & np.isfinite(y_sig) & (y_sig > 0)

    plt.figure(figsize=(12, 8))
    plt.title("ASD Comparison (ET vs. LVK.O2)", fontsize=16)


    m = band(freq_et)
    asd_net_O2 = combine_asds(asds_o2, method="network")
    plt.loglog(freq_et[m], asd_net_O2[m], lw=0.7,
               label=r"O2 network $(1/S=\sum_I 1/S_I)$")
    asd_net_et = combine_asds(asds_et, method="network")
    plt.loglog(freq_et[m], asd_net_et[m], lw=0.7,
               label=r"ET network $(1/S=\sum_I 1/S_I)$ = ET-D$/\sqrt{3}$")


    plt.loglog(freq_sig[mask_sig], y_sig[mask_sig], lw=0.7, color="crimson",
               label=f"signal (m=({m1:.1f}, {m2:.1f}), "
                     f"chirp_mass={params_sig[signal_index, 0]:.2f}, "
                     f"d_L={d_ref:.0f} Mpc)")

    # plt.axvspan(F_MIN, F_MAX, color="tab:blue", alpha=0.05,
    #             label=f"training band {F_MIN:.0f}-{F_MAX:.0f} Hz")

    plt.xlim(5.0, 2048.0)
    plt.ylim(1e-27, 1e-17)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel(r"$\sqrt{S_n(f)},\ |\tilde{h}(f)|$")
    plt.grid(True, which="both", alpha=0.15)
    plt.legend(fontsize=9)
    out_path = os.path.join(EXAMPLE_DIR, out_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved {out_path}")

    i100 = 800  # index of 100 Hz on the 0.125 Hz grid
    print(f"ASD at 100 Hz: O2 H1 = {asds_o2['H1'][i100]:.3e}, "
          f"ET1 = {asds_et['ET1'][i100]:.3e}  ->  ET is "
          f"{asds_o2['H1'][i100] / asds_et['ET1'][i100]:.1f}x lower")
    print(f"  merged ET (network) = {asd_net_et[i100]:.3e} "
          f"(= ET1/{asds_et['ET1'][i100] / asd_net_et[i100]:.3f})")
    print(f"  quadrature sum      = "
          f"{combine_asds(asds_et, method='null')[i100]:.3e} "
          f"(null stream, NOT a sensitivity)")
    print("ET1 == ET2 == ET3:",
          bool(np.array_equal(asds_et["ET1"], asds_et["ET2"]) and
               np.array_equal(asds_et["ET1"], asds_et["ET3"])))


plot_et_asd_and_signal()

print("Done.")
