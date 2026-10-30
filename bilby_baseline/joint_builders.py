from copy import deepcopy

import bilby
import h5py
import numpy as np

from builders import build_prior


BBH_EVENT_PARAMETERS = (
    "chirp_mass", "mass_ratio", "luminosity_distance", "geocent_time",
    "a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn",
    "psi", "ra", "dec", "phase",
)
_MASS_CONSTRAINTS = ("mass_1", "mass_2")
_EVENT_KEYS = frozenset(BBH_EVENT_PARAMETERS + _MASS_CONSTRAINTS)


def split_parameters(parameters):
    first = {key: parameters[key] for key in _EVENT_KEYS if key in parameters}
    second = {
        key: parameters[f"{key}_2"]
        for key in _EVENT_KEYS if f"{key}_2" in parameters
    }
    if "geocent_time" not in first or "delta_tc_2" not in parameters:
        raise KeyError("Joint parameters require geocent_time and delta_tc_2.")
    second["geocent_time"] = first["geocent_time"] - parameters["delta_tc_2"]
    return first, second


def joint_bbh_prior_conversion(sample):
    """Generate component masses separately so both BBH constraints apply."""
    first, second = split_parameters(sample)
    converted = dict(sample)
    converted["geocent_time_2"] = second["geocent_time"]
    for event, suffix in ((first, ""), (second, "_2")):
        event, _ = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
            event.copy()
        )
        event = bilby.gw.conversion.generate_mass_parameters(event)
        converted.update({f"{key}{suffix}": value for key, value in event.items()})
    return converted


def build_joint_prior(result, prior="ET", time_offset_prior=None):
    """Build a two-BBH prior using the existing single-event prior settings.

    The default offset matches train_settings_ET_JPE.yaml.  Its positive support
    fixes event 1 as the later merger, avoiding the event-label degeneracy.
    """
    base = build_prior(result, prior=prior)
    entries = {}
    for name, distribution in base.items():
        entries[name] = deepcopy(distribution)
        if name != "geocent_time":
            second = deepcopy(distribution)
            if hasattr(second, "name"):
                second.name = f"{name}_2"
            entries[f"{name}_2"] = second

    if time_offset_prior is None:
        time_offset_prior = bilby.core.prior.Uniform(
            0.01, 0.5, name="delta_tc_2"
        )
    entries["delta_tc_2"] = deepcopy(time_offset_prior)
    return bilby.core.prior.PriorDict(
        dictionary=entries, conversion_function=joint_bbh_prior_conversion
    )


class TwoSignalLikelihood(bilby.core.likelihood.Likelihood):
    """Gaussian network likelihood for the sum of two projected BBH signals.

    Time, distance, and phase are evaluated explicitly; no single-signal
    analytic marginalization is applied.  ``waveform_generator`` is the same
    single-BBH generator returned by ``builders.build_waveform_generator``.
    """

    def __init__(self, interferometers, waveform_generator):
        super().__init__()
        self.interferometers = interferometers
        self.waveform_generator = waveform_generator
        self._noise_log_likelihood = None

    def _shift_second_earlier(self, h_freq, shift_seconds):
        """Match inference.py's finite-window, zero-padded detector-strain shift."""
        h_freq = np.asarray(h_freq)
        n = 2 * (len(h_freq) - 1)
        fs = n / self.waveform_generator.duration
        ht = np.fft.irfft(h_freq, n=n)
        m = int(np.argmax(np.abs(ht)))
        contiguous = np.roll(ht, (n - 1) - m)
        k = int(round(float(shift_seconds) * fs))
        shifted = np.zeros_like(contiguous)
        if 0 < k < n:
            shifted[:n - k] = contiguous[k:]
        elif -n < k <= 0:
            s = -k
            shifted[s:] = contiguous[:n - s]
        out = np.roll(shifted, m - (n - 1))
        return np.fft.rfft(out, n=n)

    def detector_strains(self, parameters):
        """Generate and project both events, then add strains per detector."""
        first, second = split_parameters(parameters)
        # The injection generator first places both mergers at event 1's time,
        # then shifts the *projected detector strain* for event 2 in an 8 s
        # window. A geocent_time change alone is a circular Fourier shift and
        # does not reproduce the discarded samples at the window boundary.
        second_unshifted = dict(second)
        second_unshifted["geocent_time"] = first["geocent_time"]
        polarizations_1 = self.waveform_generator.frequency_domain_strain(first)
        polarizations_2 = self.waveform_generator.frequency_domain_strain(second_unshifted)
        if polarizations_1 is None or polarizations_2 is None:
            return None
        return {
            ifo.name: (
                ifo.get_detector_response(polarizations_1, first)
                + self._shift_second_earlier(
                    ifo.get_detector_response(polarizations_2, second_unshifted),
                    parameters["delta_tc_2"],
                )
            )
            for ifo in self.interferometers
        }

    def noise_log_likelihood(self):
        if self._noise_log_likelihood is None:
            self._noise_log_likelihood = -0.5 * sum(
                np.real(ifo.optimal_snr_squared(ifo.frequency_domain_strain))
                for ifo in self.interferometers
            )
        return float(self._noise_log_likelihood)

    def log_likelihood_ratio(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        if parameters is None:
            raise ValueError("Joint likelihood requires a parameter dictionary.")
        strains = self.detector_strains(parameters)
        if strains is None:
            return -np.inf
        return float(sum(
            np.real(ifo.inner_product(strains[ifo.name]))
            - 0.5 * np.real(ifo.optimal_snr_squared(strains[ifo.name]))
            for ifo in self.interferometers
        ))

    def log_likelihood(self, parameters=None):
        return self.noise_log_likelihood() + self.log_likelihood_ratio(parameters)


def build_joint_likelihood(ifos, waveform_generator):
    """Construct the two-signal likelihood from the existing Bilby builders."""
    return TwoSignalLikelihood(ifos, waveform_generator)


def read_joint_injection_parameters(result_path, t_ref):
    """Read the two injection truths in the convention used by inference.py.

    Dingo stores merger times relative to ``t_ref``.  Its overlap injection
    shifts the second detector strain earlier by the ``shift_seconds`` HDF5
    attribute, after generating the unshifted second waveform.
    """
    with h5py.File(result_path, "r") as file:
        if "injection_parameters" not in file or "injection_parameters_2" not in file:
            raise ValueError("Result must contain both injection parameter groups.")
        first = {
            name: float(dataset[()]) for name, dataset in
            file["injection_parameters"].items() if name in _EVENT_KEYS
        }
        second = {
            name: float(dataset[()]) for name, dataset in
            file["injection_parameters_2"].items() if name in _EVENT_KEYS
        }
        shift = file["injection_parameters_2"].attrs.get("shift_seconds")
        if shift is None:
            raise ValueError("Second injection has no shift_seconds attribute.")

    first_time = first["geocent_time"] + t_ref
    second_time = second["geocent_time"] + t_ref - float(shift)
    first["geocent_time"] = first_time
    second.pop("geocent_time")
    parameters = dict(first)
    parameters.update({f"{name}_2": value for name, value in second.items()})
    parameters["delta_tc_2"] = first_time - second_time
    return parameters


def check_joint_injection_likelihood(
    result_path, t_ref, likelihood, residual_tolerance=1e-2,
    loglr_tolerance=0.1,
):
    """Compare the two-waveform Bilby model with the saved noiseless injection.

    A mismatch can reveal that inference.py's finite-window linear time shift
    is not reproduced by an ordinary geocent_time shift.  Do not bypass this
    check before interpreting a joint posterior.
    """
    parameters = read_joint_injection_parameters(result_path, t_ref)
    with h5py.File(result_path, "r") as file:
        if "context/signal" not in file:
            raise ValueError("Result lacks context/signal for an injection check.")
        saved = {
            name: np.asarray(dataset) for name, dataset in
            file["context/signal"].items()
        }

    modeled = likelihood.detector_strains(parameters)
    if modeled is None:
        raise ValueError("Waveform generator returned no signal at injection truth.")
    residual = 0.0
    expected = 0.0
    for ifo in likelihood.interferometers:
        if ifo.name not in saved:
            raise KeyError(f"Saved injection has no {ifo.name} signal.")
        mask = ifo.frequency_mask
        signal = saved[ifo.name][mask]
        model = modeled[ifo.name][mask]
        data = ifo.frequency_domain_strain[mask]
        psd = ifo.power_spectral_density_array[mask]
        factor = 4.0 / ifo.duration
        residual += factor * np.sum(np.abs(model - signal) ** 2 / psd)
        expected += factor * np.sum(
            (np.real(np.conj(signal) * data) - 0.5 * np.abs(signal) ** 2) / psd
        )
    actual = likelihood.log_likelihood_ratio(parameters)
    difference = actual - expected
    if not (np.isfinite(residual) and np.isfinite(difference)
            and residual < residual_tolerance
            and abs(difference) < loglr_tolerance):
        raise ValueError(
            f"Joint injection check failed: residual={residual:.6g}, "
            f"logLR difference={difference:.6g}. Check time shifting, "
            "waveform settings, and injection metadata."
        )
    return {"residual": float(residual), "loglr_difference": float(difference)}
