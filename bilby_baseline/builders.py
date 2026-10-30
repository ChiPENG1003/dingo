import numpy as np
import bilby
import bilby.gw.source as _bilby_source

from dingo.gw.gwutils import get_interferometer


def _lal_bbh_fixed_spin_conversion_phase(
    frequency_array, mass_1, mass_2, luminosity_distance,
    a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase,
    **kwargs,
):
    spin_conversion_phase = kwargs.pop("spin_conversion_phase", None)
    if spin_conversion_phase is None:
        return _bilby_source.lal_binary_black_hole(
            frequency_array, mass_1, mass_2, luminosity_distance,
            a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs,
        )

    _orig = _bilby_source.bilby_to_lalsimulation_spins

    def _patched(*args, **kw):
        kw = dict(kw)
        kw["phase"] = float(spin_conversion_phase)
        return _orig(*args, **kw)

    _bilby_source.bilby_to_lalsimulation_spins = _patched
    try:
        return _bilby_source.lal_binary_black_hole(
            frequency_array, mass_1, mass_2, luminosity_distance,
            a_1, tilt_1, phi_12, a_2, tilt_2, phi_jl, theta_jn, phase, **kwargs,
        )
    finally:
        _bilby_source.bilby_to_lalsimulation_spins = _orig


def add_chi(parameters):
    p = parameters.copy()
    q = p["mass_ratio"] if "mass_ratio" in p else p["q"]
    a1 = p["a_1"]
    a2 = p["a_2"]
    tilt1 = p["tilt_1"]
    tilt2 = p["tilt_2"]

    chi_eff = (a1 * np.cos(tilt1) + q * a2 * np.cos(tilt2)) / (1 + q)
    p["chi_eff"] = chi_eff
    return p


def convert_to_lal_bbh_parameters_with_chi_eff(parameters):
    """Convert full component spins without reconstructing them from chi_eff."""
    return bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(
        parameters.copy()
    )


def generate_all_bbh_parameters_with_chi_eff(
    sample,
    likelihood=None,
    priors=None,
    npool=1,
):
    converted = bilby.gw.conversion.generate_all_bbh_parameters(
        sample.copy(),
        likelihood=likelihood,
        priors=priors,
        npool=npool,
    )
    converted = add_chi(converted)
    return converted


def build_ifos(result):
    domain = result.domain
    freqs = domain.sample_frequencies
    f_max = domain.f_max
    f_min = domain.f_min
    duration = 1.0 / domain.delta_f
    sampling_frequency = 2.0 * f_max
    start_time = result.t_ref - duration

    ifos = bilby.gw.detector.InterferometerList([])
    for name in result.interferometers:
        ifo = get_interferometer(name)
        ifo.minimum_frequency = f_min
        ifo.maximum_frequency = f_max

        d = np.asarray(result.context["waveform"][name], dtype=complex)
        asd = np.asarray(result.context["asds"][name], dtype=float).copy()
        band = (freqs >= f_min) & (freqs <= f_max)
        edge = asd[band][0]
        asd = np.where(band, asd, edge)
        psd = asd ** 2
        psd = np.where(np.isfinite(psd) & (psd > 0), psd, edge ** 2)

        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=freqs, psd_array=psd
        )
        ifo.set_strain_data_from_frequency_domain_strain(
            frequency_domain_strain=d,
            frequency_array=freqs,
            start_time=start_time,
        )
        ifos.append(ifo)
    return ifos, duration, sampling_frequency


def build_prior(result, prior="LVK", sample_chi_eff=False):
    if sample_chi_eff:
        raise ValueError("Sample component spins; chi_eff is derived afterwards.")
    if prior == "LVK":
        t_ref = result.t_ref

        priors = bilby.gw.prior.BBHPriorDict()

        priors["chirp_mass"] = bilby.core.prior.Uniform(10, 80, "chirp_mass")
        priors["mass_ratio"] = bilby.core.prior.Uniform(0.125, 1.0, "mass_ratio")
        priors["luminosity_distance"] = bilby.core.prior.Uniform(
            100, 2000, "luminosity_distance", "$d_L$")
        priors["geocent_time"] = bilby.core.prior.Uniform(
            t_ref - 0.1, t_ref + 0.1, "geocent_time", "$t_c$")
        priors["a_1"] = bilby.core.prior.Uniform(0, 0.88, "a_1")
        priors["a_2"] = bilby.core.prior.Uniform(0, 0.88, "a_2")
        priors["tilt_1"] = bilby.core.prior.Sine(0, np.pi, "tilt_1")
        priors["tilt_2"] = bilby.core.prior.Sine(0, np.pi, "tilt_2")
        priors["phi_12"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phi_12")
        priors["phi_jl"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phi_jl")
        priors["theta_jn"] = bilby.core.prior.Sine(0, np.pi, "theta_jn")
        priors["psi"] = bilby.core.prior.Uniform(0, 2 * np.pi, "psi")
        priors["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi, "ra")
        priors["dec"] = bilby.core.prior.Uniform(-np.pi / 2, np.pi / 2, "dec")
        priors["phase"] = bilby.core.prior.Uniform(
            0, 2 * np.pi, "phase", boundary="periodic", latex_label=r"$\phi$")

    elif prior == "ET":
        # Result resolves the checkpoint's intrinsic/extrinsic priors, but moves
        # marginalized phase/time priors out of result.prior. Restore them.
        from copy import deepcopy
        priors = deepcopy(result.prior)
        if result.phase_prior is not None:
            priors["phase"] = deepcopy(result.phase_prior)
        if result.geocent_time_prior is not None:
            priors["geocent_time"] = deepcopy(result.geocent_time_prior)
        time_prior = priors["geocent_time"]
        if not isinstance(time_prior, bilby.core.prior.Uniform):
            raise ValueError("Expected a uniform relative geocent_time prior.")
        time_prior.minimum += result.t_ref
        time_prior.maximum += result.t_ref
        # Preserve component-mass constraints and the induced spin prior.
        return bilby.gw.prior.BBHPriorDict(dictionary=dict(priors))

    else:
        raise ValueError(f"Unknown prior set: {prior}")

    # BBHPriorDict loads default component-mass constraints (5--100 Msun).
    # Those would silently truncate the chirp-mass and q ranges set above.
    max_m1 = priors["chirp_mass"].maximum * (
        (1 + priors["mass_ratio"].minimum) ** (1 / 5)
        / priors["mass_ratio"].minimum ** (3 / 5)
    )
    priors["mass_1"] = bilby.core.prior.Constraint(0, max_m1, "mass_1")
    priors["mass_2"] = bilby.core.prior.Constraint(0, max_m1, "mass_2")
    return priors


def build_waveform_generator(result, duration, sampling_frequency):
    wfg_kwargs = result.base_metadata["dataset_settings"]["waveform_generator"]

    waveform_arguments = dict(
        waveform_approximant=wfg_kwargs["approximant"],
        reference_frequency=wfg_kwargs["f_ref"],
        minimum_frequency=result.domain.f_min,
        maximum_frequency=result.domain.f_max,
        spin_conversion_phase=wfg_kwargs.get("spin_conversion_phase"),
    )

    return bilby.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=_lal_bbh_fixed_spin_conversion_phase,
        parameter_conversion=convert_to_lal_bbh_parameters_with_chi_eff,
        waveform_arguments=waveform_arguments,
    )


def build_likelihood(ifos, wfg):
    return bilby.gw.likelihood.GravitationalWaveTransient(
        interferometers=ifos,
        waveform_generator=wfg,
    )
