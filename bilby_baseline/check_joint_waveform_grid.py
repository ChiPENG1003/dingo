"""Independent Dingo/Bilby waveform checks at several joint parameter points."""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from builders import build_ifos, build_waveform_generator
from joint_builders import (
    BBH_EVENT_PARAMETERS,
    build_joint_likelihood,
    read_joint_injection_parameters,
    split_parameters,
)
from dingo.gw.injection import Injection
from dingo.gw.result import Result


def shifted_detector_strain(domain, frequency_strain, seconds):
    """Reference implementation of the injection's finite-window sample shift."""
    n = 2 * (len(frequency_strain) - 1)
    sample_shift = int(round(seconds * n * domain.delta_f))
    samples = np.fft.irfft(frequency_strain, n=n)
    peak = int(np.argmax(np.abs(samples)))
    unwrapped = np.roll(samples, n - 1 - peak)
    moved = np.zeros(n, dtype=unwrapped.dtype)
    if 0 <= sample_shift < n:
        moved[: n - sample_shift] = unwrapped[sample_shift:]
    elif -n < sample_shift < 0:
        moved[-sample_shift:] = unwrapped[: n + sample_shift]
    return np.fft.rfft(np.roll(moved, peak - (n - 1)))


def test_points(truth):
    points = [("truth", truth.copy())]
    changes = (
        ("signal1_mass", {"chirp_mass": truth["chirp_mass"] * 1.01,
                          "mass_ratio": min(0.99, truth["mass_ratio"] + 0.02)}),
        ("signal2_mass", {"chirp_mass_2": truth["chirp_mass_2"] * 1.03,
                          "mass_ratio_2": min(0.99, truth["mass_ratio_2"] + 0.02)}),
        ("sky_and_times", {"ra": (truth["ra"] + 0.2) % (2 * np.pi),
                           "dec": float(np.clip(truth["dec"] + 0.1, -1.4, 1.4)),
                           "geocent_time": truth["geocent_time"] + 0.012,
                           "delta_tc_2": truth["delta_tc_2"] + 0.013}),
        ("early_second", {"delta_tc_2": 0.35}),
        ("spin_and_distance", {"a_1": min(0.99, truth["a_1"] + 0.08),
                               "tilt_1_2": min(3.0, truth["tilt_1_2"] + 0.1),
                               "luminosity_distance_2": truth["luminosity_distance_2"] * 1.1}),
    )
    for name, updates in changes:
        points.append((name, {**truth, **updates}))
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = Result(file_name=args.result)
    ifos, duration, sampling_frequency = build_ifos(result)
    likelihood = build_joint_likelihood(
        ifos, build_waveform_generator(result, duration, sampling_frequency)
    )
    truth = read_joint_injection_parameters(args.result, result.t_ref)
    injection = Injection.from_posterior_model_metadata(result.base_metadata)
    injection.whiten = False

    report = []
    for name, parameters in test_points(truth):
        first, second = split_parameters(parameters)
        first = {k: v for k, v in first.items() if k in BBH_EVENT_PARAMETERS}
        second = {k: v for k, v in second.items() if k in BBH_EVENT_PARAMETERS}
        first["geocent_time"] -= result.t_ref
        second["geocent_time"] = first["geocent_time"]
        dingo1 = injection.signal(first)["waveform"]
        dingo2 = injection.signal(second)["waveform"]
        model = likelihood.detector_strains(parameters)
        if model is None:
            raise ValueError(f"{name}: Bilby waveform generation failed")
        squared_error = 0.0
        model_norm = 0.0
        direct_loglr = 0.0
        for ifo in ifos:
            expected = dingo1[ifo.name] + shifted_detector_strain(
                injection.data_domain, dingo2[ifo.name],
                parameters["delta_tc_2"],
            )
            mask = ifo.frequency_mask
            psd = ifo.power_spectral_density_array[mask]
            norm = 4.0 / ifo.duration
            difference = model[ifo.name][mask] - expected[mask]
            squared_error += norm * np.sum(np.abs(difference) ** 2 / psd)
            model_norm += norm * np.sum(np.abs(expected[mask]) ** 2 / psd)
            data = ifo.frequency_domain_strain[mask]
            direct_loglr += norm * np.sum(
                (np.real(np.conj(data) * expected[mask])
                 - 0.5 * np.abs(expected[mask]) ** 2) / psd
            )
        actual_loglr = likelihood.log_likelihood_ratio(parameters)
        row = {
            "point": name,
            "whitened_residual_squared": float(squared_error),
            "relative_waveform_error": float(np.sqrt(squared_error / model_norm)),
            "loglr_difference": float(actual_loglr - direct_loglr),
        }
        report.append(row)
        print(row, flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as file:
        json.dump(report, file, indent=2)
    if any(row["relative_waveform_error"] > 1e-4
           or abs(row["loglr_difference"]) > 0.1 for row in report):
        raise SystemExit("Multi-point waveform check failed; sampler was not started")


if __name__ == "__main__":
    main()
