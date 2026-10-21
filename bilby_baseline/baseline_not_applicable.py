import bilby
import numpy as np
import os
import h5py
from dingo.gw.result import Result

from bilby_baseline.builders import (
    build_ifos,
    build_waveform_generator,
    build_likelihood,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

bilby.core.utils.random.seed(2026)

outdir = 'gw_example'
label = 'binary_bh'

bilby.utils.setup_logger(outdir=outdir, label=label)

# ==============================
# Load the Dingo injection: the SAME data + settings Dingo used for inference.
# context/waveform = strain (signal+noise), context/asds = PSD. NOT re-injected.
# ==============================
injection_file = os.path.join(SCRIPT_DIR, f"injection.hdf5")
result = Result(file_name=injection_file)

# Injection truth, embedded under /injection_parameters.
with h5py.File(injection_file, "r") as f:
    injection_parameters = {
        key: float(f["injection_parameters"][key][()])
        for key in f["injection_parameters"].keys()
    }

# Dingo stores geocent_time RELATIVE to t_ref; bilby needs absolute GPS.
t_ref = result.t_ref
injection_parameters["geocent_time"] = t_ref + injection_parameters["geocent_time"]

# ==============================
# Interferometers (H1 + L1), waveform generator, likelihood — all built from the
# real data / Dingo settings (IMRPhenomXPHM, f_ref=20, f_min=20, f_max=1024, fs=2048).
# ==============================
ifos, duration, sampling_frequency = build_ifos(result)
waveform_generator = build_waveform_generator(result, duration, sampling_frequency)
likelihood = build_likelihood(ifos, waveform_generator)

minimum_frequency = result.domain.f_min

# ==============================
# Define the prior　distributions
# ==============================
priors = bilby.gw.prior.BBHPriorDict()

priors["chirp_mass"] = bilby.core.prior.Uniform(15, 100, "chirp_mass")
priors["mass_ratio"] = bilby.core.prior.Uniform(0.2, 1.0, "mass_ratio")
priors["luminosity_distance"] = bilby.core.prior.Uniform(100, 500, "luminosity_distance")
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 0.1,
    injection_parameters["geocent_time"] + 0.1,
    "geocent_time",
)
priors["a_1"] = bilby.core.prior.Uniform(0, 0.99, "a_1")
priors["a_2"] = bilby.core.prior.Uniform(0, 0.99, "a_2")
priors["tilt_1"] = bilby.core.prior.Sine(0, np.pi, "tilt_1")
priors["tilt_2"] = bilby.core.prior.Sine(0, np.pi, "tilt_2")
priors["phi_12"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phi_12")
priors["phi_jl"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phi_jl")
priors["theta_jn"] = bilby.core.prior.Sine(0, np.pi, "theta_jn")
priors["psi"] = bilby.core.prior.Uniform(0, 2 * np.pi, "psi")
priors["ra"] = bilby.core.prior.Uniform(0, 2 * np.pi, "ra")
priors["dec"] = bilby.core.prior.Uniform(-np.pi / 2, np.pi / 2, "dec")
priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, "phase")

free_parameters = {
    "chirp_mass",
    "mass_ratio",
    "luminosity_distance",
    "geocent_time",
    "a_1",
    "a_2",
    "tilt_1",
    "tilt_2",
    "phi_12",
    "phi_jl",
    "theta_jn",
    "psi",
    "ra",
    "dec",
    "phase"
}

for key in list(priors.keys()):
    if key in free_parameters:
        continue
    if isinstance(priors[key], bilby.core.prior.Constraint):
        continue  # keep mass_1/mass_2 as constraints derived from chirp_mass/mass_ratio
    priors[key] = injection_parameters[key]

print("has redundant key?", priors.test_has_redundant_keys())

priors.validate_prior(
    duration = duration,
    minimum_frequency = minimum_frequency,
)

# ==============================
# Run the sampler
# ==============================
result = bilby.run_sampler(
    likelihood = likelihood,
    priors = priors,
    sampler = "dynesty",
    nlive = 250,
    outdir = outdir,
    label = label,
    injection_parameters = injection_parameters,
    plot= True,
    resume = True,
    clean = True,
)

# ==============================
# Print a short summary
# ==============================
print(result)
print(
    result.posterior[
        ["chirp_mass", "mass_ratio", "luminosity_distance", "geocent_time", "a_1", "a_2",
          "tilt_1", "tilt_2", "phi_12", "phi_jl", "theta_jn", "psi","ra", "dec", "phase"]
    ].describe()
)

# ==============================
# Produce a corner plot
# ==============================
result.plot_corner()
