"""
Transforms for training on overlapping signals (joint posterior estimation).

The dataset stores the two signals separately: signal 1 under the usual keys
(``h_plus``, ``h_cross`` and unsuffixed parameters) and signal 2 under suffixed ones
(``h_plus_2``, ``h_cross_2``, ``chirp_mass_2``, ...). Nothing about the two signals is
shared: each gets its own sky position, polarization angle, luminosity distance and
merger time, drawn independently at training time. They are combined only after both
have been projected onto the detectors, which is where two overlapping sources
actually add.

The ordering convention is that signal 1 merges *last*. Signal 2's merger time is
parameterized by a strictly positive offset ``delta_tc_2`` (signal 2 merges
``delta_tc_2`` seconds earlier), which breaks the permutation symmetry between the two
sources. Without it the likelihood is invariant under exchanging the two signals and
the flow has to model a bimodal posterior with one mode per labelling -- the failure
mode reported for joint estimation with normalizing flows (arXiv:2211.15097). Because
``delta_tc_2`` is a time *difference*, it is also invariant under the global time shift
that GNPE applies, so it can be used as an inference parameter alongside GNPE.
"""

import numpy as np

from dingo.gw.prior import BBHExtrinsicPriorDict
from dingo.gw.transforms.detector_transforms import (
    GetDetectorTimes,
    ProjectOntoDetectors,
)
from dingo.gw.transforms.utils import get_batch_size_of_input_sample

# Extrinsic parameters that each signal owns a private copy of.
SIGNAL_EXTRINSIC_PARAMETERS = ["ra", "dec", "psi", "luminosity_distance", "geocent_time"]


def finite_window_time_shift(domain, frequency_strain, seconds):
    """Move a merger earlier by ``seconds`` without wrapping strain across the 8 s window.

    This is the vectorized version of the overlap injection's ``_linear_time_shift``.
    The last axis holds the rFFT bins; any preceding axes are batch dimensions.
    """
    frequency_strain = np.asarray(frequency_strain)
    n = 2 * (frequency_strain.shape[-1] - 1)
    samples = np.fft.irfft(frequency_strain, n=n, axis=-1)
    shift = np.broadcast_to(
        np.rint(np.asarray(seconds) * n * domain.delta_f).astype(int),
        samples.shape[:-1],
    )
    peak = np.argmax(np.abs(samples), axis=-1)
    indices = np.arange(n)
    contiguous = np.take_along_axis(
        samples, (indices + np.expand_dims(peak, -1) + 1) % n, axis=-1
    )
    source = indices + np.expand_dims(shift, -1)
    moved = np.where(
        (source >= 0) & (source < n),
        np.take_along_axis(contiguous, np.clip(source, 0, n - 1), axis=-1),
        0,
    )
    restored = np.take_along_axis(
        moved, (indices + n - 1 - np.expand_dims(peak, -1)) % n, axis=-1
    )
    return np.fft.rfft(restored, n=n, axis=-1)


class SampleExtrinsicParametersForSignal(object):
    """
    Sample an independent set of extrinsic parameters for an additional signal.

    The sampled values are written to sample['extrinsic_parameters'] under
    ``<name> + suffix``, leaving the first signal's parameters untouched.

    If `time_offset_prior` is given, geocent_time is not drawn from the extrinsic
    prior. Instead an offset ``delta_tc<suffix>`` is drawn from it and the signal is
    placed that many seconds *before* signal 1. A prior with positive support therefore
    fixes the ordering of the two signals; see the module docstring.
    """

    def __init__(self, extrinsic_prior_dict, suffix="_2", time_offset_prior=None):
        self.extrinsic_prior_dict = extrinsic_prior_dict
        self.suffix = suffix
        self.prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
        self.time_offset_prior = time_offset_prior
        if time_offset_prior is not None:
            self.offset_prior = BBHExtrinsicPriorDict(
                {f"delta_tc{suffix}": time_offset_prior}
            )
        else:
            self.offset_prior = None

    def __call__(self, input_sample):
        sample = input_sample.copy()
        batched, batch_size = get_batch_size_of_input_sample(input_sample)
        n = batch_size if batched else None

        new = self.prior.sample(n)
        new = {
            k: v.astype(np.float32) if batched else float(v) for k, v in new.items()
        }
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        if self.offset_prior is not None:
            offset = self.offset_prior.sample(n)[f"delta_tc{self.suffix}"]
            offset = offset.astype(np.float32) if batched else float(offset)
            new["geocent_time"] = extrinsic_parameters["geocent_time"] - offset
            # Recorded as an inference parameter: unlike geocent_time_2 it is a time
            # difference, so it survives the global time shift imposed by GNPE.
            extrinsic_parameters[f"delta_tc{self.suffix}"] = offset

        for k, v in new.items():
            extrinsic_parameters[k + self.suffix] = v
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample

    @property
    def reproduction_dict(self):
        return {
            "extrinsic_prior_dict": self.extrinsic_prior_dict,
            "suffix": self.suffix,
            "time_offset_prior": self.time_offset_prior,
        }


class GetDetectorTimesForSignal(object):
    """
    Compute detector coalescence times for an additional signal.

    Same as GetDetectorTimes, but reads ``ra<suffix>``, ``dec<suffix>``,
    ``geocent_time<suffix>`` and writes ``<ifo><suffix>_time``.
    """

    def __init__(self, ifo_list, ref_time, suffix="_2"):
        self.ifo_list = ifo_list
        self.ref_time = ref_time
        self.suffix = suffix
        self._inner = GetDetectorTimes(ifo_list, ref_time)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        view = {
            k: extrinsic_parameters[k + self.suffix]
            for k in ("ra", "dec", "geocent_time")
        }
        out = self._inner({"extrinsic_parameters": view})["extrinsic_parameters"]
        for ifo in self.ifo_list:
            extrinsic_parameters[f"{ifo.name}{self.suffix}_time"] = out[
                f"{ifo.name}_time"
            ]
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ApplyGNPEShiftToSignal(object):
    """
    Carry the GNPE time shift over to an additional signal.

    GNPE time-shifts the *data*, so the shift has to be applied to every signal in it.
    GNPECoalescenceTimes only adjusts signal 1's detector times; this subtracts the
    same per-detector proxies from ``<ifo><suffix>_time`` so that the relative timing
    of the two signals is preserved. Without it, GNPE would move signal 1 and leave
    signal 2 in place, corrupting delta_tc.

    Must run after GNPECoalescenceTimes and before the projection.
    """

    def __init__(self, ifo_list, suffix="_2"):
        self.ifo_list = ifo_list
        self.suffix = suffix

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        for ifo in self.ifo_list:
            proxy = extrinsic_parameters[f"{ifo.name}_time_proxy"]
            key = f"{ifo.name}{self.suffix}_time"
            extrinsic_parameters[key] = extrinsic_parameters[key] - proxy
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ProjectOntoDetectorsAndSum(object):
    """
    Project each signal onto the detectors with its own extrinsic parameters, then add
    the resulting strains. A signal with ``delta_tc<suffix>`` is first projected at
    its unshifted reference time and then moved earlier with the exact finite-window
    convention used to build the overlap injections.

    This is the point at which the overlap is formed. Summing here rather than at the
    polarization level is what lets the two sources have different sky positions,
    polarization angles and distances: the antenna patterns and the distance rescaling
    are applied per signal, before the sum.

    Parameters
    ----------
    ifo_list : InterferometerList
    domain : Domain
    ref_time : float
    suffixes : sequence of str
        One entry per signal. "" is signal 1 (unsuffixed keys), "_2" the second, and so
        on. The polarizations of signal k are read from ``h_plus<suffix>`` /
        ``h_cross<suffix>``, its stored reference parameters from
        ``luminosity_distance<suffix>`` / ``geocent_time<suffix>``, and its sampled
        extrinsic parameters from ``ra<suffix>`` etc.
    """

    def __init__(self, ifo_list, domain, ref_time, suffixes=("", "_2")):
        if "" not in suffixes:
            raise ValueError('suffixes must contain "" (the first signal).')
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time
        self.suffixes = tuple(suffixes)
        self._project = ProjectOntoDetectors(ifo_list, domain, ref_time)

    def __call__(self, input_sample):
        sample = input_sample.copy()
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        strains = None
        for suffix in self.suffixes:
            offset = extrinsic_parameters.get(f"delta_tc{suffix}") if suffix else None
            # The stored second-signal time already includes -offset. Undo it for
            # projection, then apply the finite-window shift after projection.
            # This also preserves the GNPE proxy applied to detector times above.
            sub = {
                "waveform": {
                    pol: sample["waveform"][pol + suffix]
                    for pol in ("h_plus", "h_cross")
                },
                # ProjectOntoDetectors reads the reference distance and the (zero)
                # fiducial geocent_time from 'parameters', and pops the sampled
                # extrinsic parameters. Both are private to this signal.
                "parameters": {
                    k: parameters[k + suffix]
                    for k in ("luminosity_distance", "geocent_time")
                },
                "extrinsic_parameters": {
                    **{k: (extrinsic_parameters[k + suffix] + offset
                           if offset is not None and k == "geocent_time"
                           else extrinsic_parameters[k + suffix])
                       for k in SIGNAL_EXTRINSIC_PARAMETERS},
                    **{f"{ifo.name}_time":
                       (extrinsic_parameters[f"{ifo.name}{suffix}_time"] + offset
                        if offset is not None
                        else extrinsic_parameters[f"{ifo.name}{suffix}_time"])
                       for ifo in self.ifo_list},
                },
            }
            out = self._project(sub)
            if offset is not None:
                out["waveform"] = {
                    name: finite_window_time_shift(self.domain, strain, offset)
                    for name, strain in out["waveform"].items()
                }
                out["parameters"]["geocent_time"] -= offset
                for ifo in self.ifo_list:
                    out["parameters"][f"{ifo.name}_time"] -= offset
            if strains is None:
                strains = out["waveform"]
            else:
                strains = {k: v + out["waveform"][k] for k, v in strains.items()}

            # Record the realised extrinsic values of this signal, and drop the ones
            # the projection consumed so they do not linger in two places. Detector
            # times keep the "<ifo><suffix>_time" spelling used everywhere else,
            # rather than the "<ifo>_time<suffix>" the inner transform would give.
            ifo_times = {
                f"{ifo.name}{suffix}_time": out["parameters"].pop(f"{ifo.name}_time")
                for ifo in self.ifo_list
            }
            for k, v in out["parameters"].items():
                parameters[k + suffix] = v
            parameters.update(ifo_times)
            for k in SIGNAL_EXTRINSIC_PARAMETERS:
                extrinsic_parameters.pop(k + suffix, None)
            for ifo in self.ifo_list:
                extrinsic_parameters.pop(f"{ifo.name}{suffix}_time", None)

        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample
