from typing import Optional
import numpy as np
from copy import deepcopy
from scipy.signal.windows import tukey
from scipy.interpolate import interp1d

from bilby.gw.detector import (
    InterferometerList,
    PowerSpectralDensity,
    TriangularInterferometer,
    get_empty_interferometer,
)

from dingo.gw.prior import default_extrinsic_dict
from dingo.gw.prior import BBHExtrinsicPriorDict


def get_interferometer(name):
    """
    Return the bilby Interferometer with the given name.

    This extends bilby.gw.detector.get_empty_interferometer to the individual
    interferometers of triangular detectors. bilby only ships the parent detector
    ('ET'), which it instantiates as a TriangularInterferometer holding the three
    nested interferometers 'ET1', 'ET2', 'ET3'. Since dingo keys strain data, ASDs
    and GNPE proxies by the individual interferometer names, these have to be
    addressable one by one.

    Parameters
    ----------
    name : str
        Detector name, e.g. 'H1' or 'ET1'.

    Returns
    -------
    Interferometer
    """
    try:
        return get_empty_interferometer(name)
    except ValueError:
        parent_name = name.rstrip("0123456789")
        if not parent_name or parent_name == name:
            raise
        try:
            parent = get_empty_interferometer(parent_name)
        except ValueError:
            raise ValueError(f"Interferometer {name} not implemented")
        if not isinstance(parent, TriangularInterferometer):
            raise ValueError(f"Interferometer {name} not implemented")
        for ifo in parent:
            if ifo.name == name:
                return ifo
        raise ValueError(
            f"Interferometer {name} not implemented. The triangular detector "
            f"{parent_name} consists of {[ifo.name for ifo in parent]}."
        )


def get_ifo_list(detectors):
    """
    Build a bilby InterferometerList from a list of detector names.

    Compared to instantiating InterferometerList directly, this also accepts the
    individual interferometers of triangular detectors (e.g. ['ET1', 'ET2', 'ET3']);
    see get_interferometer. Note that passing the parent name ('ET') also yields the
    three interferometers, but leaves the detector names in the settings out of sync
    with the keys of the strain and ASD dicts, so the individual names should be used.

    Parameters
    ----------
    detectors : list
        Detector names. Interferometer objects are passed through unchanged.

    Returns
    -------
    InterferometerList
    """
    return InterferometerList(
        [get_interferometer(d) if isinstance(d, str) else d for d in detectors]
    )


def get_window(window_kwargs):
    """Compute window from window_kwargs."""
    type = window_kwargs["type"]
    if type == "tukey":
        roll_off, T, f_s = (
            window_kwargs["roll_off"],
            window_kwargs["T"],
            window_kwargs["f_s"],
        )
        alpha = 2 * roll_off / T
        w = tukey(int(T * f_s), alpha)
        return w
    else:
        raise NotImplementedError(f"Unknown window type {type}.")


def get_extrinsic_prior_dict(extrinsic_prior):
    """Build dict for extrinsic prior by starting with
    default_extrinsic_dict, and overwriting every element for which
    extrinsic_prior is not default.
    TODO: Move to dingo.gw.prior.py?"""
    extrinsic_prior_dict = default_extrinsic_dict.copy()
    for k, v in extrinsic_prior.items():
        if v.lower() != "default":
            extrinsic_prior_dict[k] = v
    return extrinsic_prior_dict


def get_mismatch(a, b, domain, asd_file=None):
    """
    Mistmatch is 1 - overlap, where overlap is defined by
    inner(a, b) / sqrt(inner(a, a) * inner(b, b)).
    See e.g. Eq. (44) in https://arxiv.org/pdf/1106.1021.pdf.

    Parameters
    ----------
    a
    b
    domain
    asd_file

    Returns
    -------

    """
    if asd_file is not None:
        # whiten a and b, such that we can use flat-spectrum inner products below
        psd = PowerSpectralDensity(asd_file=asd_file)
        asd_interp = interp1d(
            psd.frequency_array, psd.asd_array, bounds_error=False, fill_value=np.inf
        )
        asd_array = asd_interp(domain.sample_frequencies)
        a = a / asd_array
        b = b / asd_array
    min_idx = domain.min_idx
    inner_ab = np.sum((a.conj() * b)[..., min_idx:], axis=-1).real
    inner_aa = np.sum((a.conj() * a)[..., min_idx:], axis=-1).real
    inner_bb = np.sum((b.conj() * b)[..., min_idx:], axis=-1).real
    overlap = inner_ab / np.sqrt(inner_aa * inner_bb)
    return 1 - overlap


def get_standardization_dict(
    extrinsic_prior_dict,
    wfd,
    selected_parameters,
    transform=None,
    extrinsic_suffixes=(),
    extrinsic_prior_dict_suffixed=None,
):
    """
    Calculates the mean and standard deviation of parameters. This is needed for
    standardizing neural-network input and output.

    Parameters
    ----------
    extrinsic_prior_dict : dict
    wfd : WaveformDataset
    selected_parameters : list[str]
        List of parameters for which to estimate standardization factors.
    transform : Transform
        Operator that will generate samples for parameters contained in
        selected_parameters that are not contained in the intrinsic or extrinsic prior.
        (E.g., H1_time, L1_time_proxy)
    extrinsic_suffixes : sequence of str
        Suffixes of additional overlapping signals (e.g. ("_2",)). Each of these
        signals draws its extrinsic parameters from its own prior, so
        e.g. luminosity_distance_2 is standardized with the extrinsic prior rather
        than with the fiducial value stored in the waveform dataset (whose standard
        deviation is zero).
    extrinsic_prior_dict_suffixed : dict
        Extrinsic prior of the additional signals. Defaults to extrinsic_prior_dict.

    Returns
    -------

    """
    # The intrinsic standardization is estimated based on the entire dataset.
    mean_intrinsic, std_intrinsic = wfd.parameter_mean_std()

    # Some of the extrinsic prior parameters have analytic means and standard
    # deviations. If possible, this will either get these, or else it will estimate
    # them numerically.
    ext_prior = BBHExtrinsicPriorDict(extrinsic_prior_dict)
    mean_extrinsic, std_extrinsic = ext_prior.mean_std(ext_prior.keys())

    # Check that overlap between intrinsic and extrinsic parameters is only
    # due to fiducial values (-> std 0)
    for k in std_intrinsic.keys() & std_extrinsic.keys():
        if std_intrinsic[k] != 0:
            raise ValueError(
                f"Expected intrinsic prior for {k} to be a fixed value in the waveform dataset, "
                f"since {k} is specified as an extrinsic prior in the train settings and will be sampled"
                f"during training. However, the standard deviation of {k} is non-zero: {std_intrinsic[k]}"
                f"Please re-generate the waveform dataset with a fixed value for {k}."
            )

    # Merge dicts, overwriting fiducial values for parameters (e.g.,
    # luminosity_distance) in intrinsic parameters by the extrinsic ones
    mean = {**mean_intrinsic, **mean_extrinsic}
    std = {**std_intrinsic, **std_extrinsic}

    # Additional overlapping signals sample their own extrinsic parameters, so the
    # suffixed names standardize like the extrinsic prior, not like the fiducial
    # values stored in the waveform dataset. Without this, e.g.
    # luminosity_distance_2 would inherit the stored constant and have std 0.
    if extrinsic_suffixes:
        suffixed_prior = BBHExtrinsicPriorDict(
            extrinsic_prior_dict_suffixed
            if extrinsic_prior_dict_suffixed is not None
            else extrinsic_prior_dict
        )
        mean_s, std_s = suffixed_prior.mean_std(suffixed_prior.keys())
        for suffix in extrinsic_suffixes:
            for k in mean_s:
                mean[k + suffix] = mean_s[k]
                std[k + suffix] = std_s[k]

    # For all remaining parameters that require standardization, we use the transform
    # to sample these and estimate the mean and standard deviation numerically.
    additional_parameters = [p for p in selected_parameters if p not in mean]
    if additional_parameters:
        num_samples = min(100_000, len(wfd.parameters))
        samples = {p: np.empty(num_samples) for p in additional_parameters}
        for n in range(num_samples):
            sample = {"parameters": wfd.parameters.iloc[n].to_dict()}
            sample = transform(sample)
            for p in additional_parameters:
                # This assumes all of the additional parameters are contained within
                # extrinsic_parameters. We have set it up so this is the case for the
                # GNPE proxies and the detector coalescence times.
                samples[p][n] = sample["extrinsic_parameters"][p]
        mean_additional = {p: np.mean(samples[p]).item() for p in additional_parameters}
        std_additional = {p: np.std(samples[p]).item() for p in additional_parameters}

        mean.update(mean_additional)
        std.update(std_additional)

    standardization_dict = {
        "mean": {k: mean[k] for k in selected_parameters},
        "std": {k: std[k] for k in selected_parameters},
    }
    return standardization_dict
