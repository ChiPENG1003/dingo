"""Check that JPE training superposition matches an existing overlap injection."""

import argparse
import json
import os
import sys

import numpy as np

from dingo.gw.injection import Injection
from dingo.gw.result import Result
from dingo.gw.transforms.detector_transforms import GetDetectorTimes
from dingo.gw.transforms.overlap_transforms import (
    GetDetectorTimesForSignal,
    ProjectOntoDetectorsAndSum,
    finite_window_time_shift,
)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bilby_baseline"))
from builders import build_ifos
from joint_builders import (
    BBH_EVENT_PARAMETERS,
    read_joint_injection_parameters,
    split_parameters,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    result = Result(file_name=args.result)
    truth = read_joint_injection_parameters(args.result, result.t_ref)
    ifos, _, _ = build_ifos(result)
    generator = Injection.from_posterior_model_metadata(result.base_metadata)
    generator.whiten = False
    domain = generator.data_domain
    detector_times_1 = GetDetectorTimes(generator.ifo_list, result.t_ref)
    detector_times_2 = GetDetectorTimesForSignal(generator.ifo_list, result.t_ref)
    joint_project = ProjectOntoDetectorsAndSum(
        generator.ifo_list, domain, result.t_ref
    )

    rows = []
    for name, delta in (("truth", truth["delta_tc_2"]), ("offset_0p35", 0.35)):
        parameters = {**truth, "delta_tc_2": delta}
        first, second = split_parameters(parameters)
        first = {k: float(v) for k, v in first.items() if k in BBH_EVENT_PARAMETERS}
        second = {k: float(v) for k, v in second.items() if k in BBH_EVENT_PARAMETERS}
        first["geocent_time"] -= result.t_ref
        second["geocent_time"] = first["geocent_time"]

        signals = [generator.signal(first), generator.signal(second)]
        waveform = {}
        for suffix, theta in (("", first), ("_2", second)):
            intrinsic = {
                k: v for k, v in theta.items()
                if k not in ("geocent_time", "luminosity_distance", "ra", "dec", "psi")
            }
            intrinsic["geocent_time"] = 0.0
            intrinsic["luminosity_distance"] = 100.0
            polarizations = generator.waveform_generator.generate_hplus_hcross(intrinsic)
            for key, value in polarizations.items():
                waveform[key + suffix] = domain.update_data(value)

        extrinsic = {}
        for key in ("geocent_time", "luminosity_distance", "ra", "dec", "psi"):
            extrinsic[key] = first[key]
            extrinsic[key + "_2"] = second[key]
        extrinsic["geocent_time_2"] -= delta
        extrinsic["delta_tc_2"] = delta
        sample = {
            "waveform": waveform,
            "parameters": {
                "geocent_time": 0.0,
                "luminosity_distance": 100.0,
                "geocent_time_2": 0.0,
                "luminosity_distance_2": 100.0,
            },
            "extrinsic_parameters": extrinsic,
        }
        sample = detector_times_1(sample)
        sample = detector_times_2(sample)
        training = joint_project(sample)["waveform"]

        error = norm = 0.0
        for ifo in ifos:
            expected = signals[0]["waveform"][ifo.name] + finite_window_time_shift(
                domain, signals[1]["waveform"][ifo.name], delta
            )
            mask = ifo.frequency_mask
            psd = ifo.power_spectral_density_array[mask]
            factor = 4.0 / ifo.duration
            error += factor * np.sum(abs(training[ifo.name][mask] - expected[mask]) ** 2 / psd)
            norm += factor * np.sum(abs(expected[mask]) ** 2 / psd)
        row = {"point": name, "whitened_residual_squared": float(error),
               "relative_waveform_error": float(np.sqrt(error / norm))}
        print(row, flush=True)
        rows.append(row)

    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    if any(row["relative_waveform_error"] > 1e-4 for row in rows):
        raise SystemExit("JPE training waveform does not match overlap injection")


if __name__ == "__main__":
    main()
