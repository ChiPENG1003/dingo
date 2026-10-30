"""Shared plumbing for the overlapping-injection parameter grids.

Everything that touches the network, the prior or the strain is imported from
inference.py rather than reimplemented, so a change there propagates here. This
module owns only what inference.py has no reason to know about: the mass
constraint algebra, the SNR bookkeeping, and the grid's own paths.

Nothing in inference.py is modified by this package.
"""

import json
import os
import sys
import types

import numpy as np

COMBO_DIR = os.path.dirname(os.path.abspath(__file__))
INFERENCE_DIR = os.path.dirname(COMBO_DIR)
EXAMPLE_DIR = os.path.dirname(INFERENCE_DIR)
CONSTRUCT_DIR = os.path.join(EXAMPLE_DIR, "construct_plot_data")

for _p in (INFERENCE_DIR, CONSTRUCT_DIR, COMBO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ruff: noqa: E402  -- the sys.path setup above has to run before these imports.
import inference as inf
from construct_data import compute_optimal_snr

RESULTS_DIR = os.path.join(COMBO_DIR, "results")
REFERENCE_DIR = os.path.join(RESULTS_DIR, "reference")

# The band the SNR is defined over. Matches the model domain (f_max = 1024,
# delta_f = 0.125); read back from the domain at runtime rather than trusted.
F_MIN = 20.0
F_MAX = 1024.0

# --------------------------------------------------------------------------
# The y axis, shared by every sweep: 10 log-spaced time separations.
# --------------------------------------------------------------------------

TIME_SHIFTS = np.geomspace(0.05, 1.0, 10)

# _linear_time_shift() moves signal 2 EARLIER for a positive shift: it rolls the
# merger to the last sample, drops the first k samples, and rolls back, so the
# merger index goes m -> m - k. (inference.py's --shift help text says "later";
# the code and its own inline comment say earlier. verify_shift_direction()
# below measures it, and select_target.py runs that check.) The grid only ever
# uses positive shifts, so the axis is really |Δt_merger| in seconds.


def verify_shift_direction(injection_generator, theta, shift=0.5):
    """Measure which way _linear_time_shift moves a signal. Returns seconds by
    which the merger moved LATER (so a negative number means earlier)."""
    domain = injection_generator.data_domain
    h = injection_generator.signal(theta)["waveform"]
    det = sorted(h)[0]
    n = 2 * (len(h[det]) - 1)
    fs = n * domain.delta_f

    def _peak_time(h_freq):
        return int(np.argmax(np.abs(np.fft.irfft(np.asarray(h_freq), n=n)))) / fs

    t0 = _peak_time(h[det])
    t1 = _peak_time(inf._linear_time_shift(domain, h[det], shift))
    # The buffer is periodic, so unwrap the difference onto (-T/2, T/2].
    period = n / fs
    return (t1 - t0 + period / 2) % period - period / 2


# --------------------------------------------------------------------------
# Mass algebra. The prior puts Constraint(10, 80) on BOTH component masses,
# which makes large parts of the (chirp_mass, mass_ratio) rectangle empty:
# equal masses at m=80 cap the chirp mass at 80 * 2^(-1/5) = 69.64, and
# q = 0.125 is reachable only at the single point (m1, m2) = (80, 10).
# --------------------------------------------------------------------------

MASS_MIN = 10.0
MASS_MAX = 80.0


def component_masses(chirp_mass, mass_ratio):
    """(m1, m2) from (chirp_mass, q = m2/m1 <= 1)."""
    q = np.asarray(mass_ratio, dtype=float)
    eta = q / (1.0 + q) ** 2
    total = np.asarray(chirp_mass, dtype=float) * eta ** (-0.6)
    return total / (1.0 + q), total * q / (1.0 + q)


def masses_allowed(chirp_mass, mass_ratio, m_min=MASS_MIN, m_max=MASS_MAX):
    m1, m2 = component_masses(chirp_mass, mass_ratio)
    return (m1 >= m_min) & (m1 <= m_max) & (m2 >= m_min) & (m2 <= m_max)


def q_range_for_chirp_mass(chirp_mass, q_min=0.125, q_max=1.0, n=20001):
    """Interval of mass ratios compatible with a fixed chirp mass, or None.

    Scanned rather than solved: the boundary is a root of a quintic in q^(1/5)
    and a 2e4-point scan is exact to ~4e-5 in q, far below anything that matters
    for a 10-bin axis.
    """
    q = np.linspace(q_min, q_max, n)
    ok = masses_allowed(chirp_mass, q)
    if not ok.any():
        return None
    return float(q[ok].min()), float(q[ok].max())


def chirp_mass_range_for_q(mass_ratio, mc_min=10.0, mc_max=80.0, n=20001):
    """Interval of chirp masses compatible with a fixed mass ratio, or None."""
    mc = np.linspace(mc_min, mc_max, n)
    ok = masses_allowed(mc, mass_ratio)
    if not ok.any():
        return None
    return float(mc[ok].min()), float(mc[ok].max())


def max_chirp_mass_for_q(mass_ratio):
    rng = chirp_mass_range_for_q(mass_ratio)
    return None if rng is None else rng[1]


def sample_truncated(prior, lo, hi, rng):
    """Draw from a bilby prior restricted to [lo, hi], keeping its shape.

    Inverse-CDF where the prior supports it (UniformInComponentsChirpMass and
    UniformInComponentsMassRatio are both analytic), else rejection sampling.
    Rejection alone would be unusable here: at q = 0.15 the admissible chirp
    mass window is 4 Msun wide out of a 70 Msun prior.
    """
    lo = max(float(lo), float(prior.minimum))
    hi = min(float(hi), float(prior.maximum))
    if not hi > lo:
        return float(lo)
    try:
        c_lo, c_hi = float(prior.cdf(lo)), float(prior.cdf(hi))
        if np.isfinite(c_lo) and np.isfinite(c_hi) and c_hi > c_lo:
            return float(prior.rescale(rng.uniform(c_lo, c_hi)))
    except (AttributeError, NotImplementedError, ValueError):
        pass
    for _ in range(10000):
        value = float(prior.sample())
        if lo <= value <= hi:
            return value
    raise RuntimeError(f"Could not sample {prior} inside [{lo}, {hi}].")


# --------------------------------------------------------------------------
# SNR. Always computed on the PURE signal -- context/waveform is d = h + n and
# gives sqrt(2 N_bins) ~ 126 regardless of distance.
# --------------------------------------------------------------------------

def detector_snrs(waveform, asds, delta_f, f_min=F_MIN, f_max=F_MAX):
    return {
        det: float(compute_optimal_snr(h, asds[det], delta_f, fmin=f_min, fmax=f_max))
        for det, h in waveform.items()
    }


def network_snr(waveform, asds, delta_f, f_min=F_MIN, f_max=F_MAX):
    """Detectors are independent, so per-detector SNRs add in quadrature --
    the same convention as the published GWTC network SNRs."""
    snrs = detector_snrs(waveform, asds, delta_f, f_min=f_min, f_max=f_max)
    return float(np.sqrt(sum(s**2 for s in snrs.values()))), snrs


def projected_signal(injection_generator, theta, shift_seconds=None):
    """Signal 2 as it will actually be injected: projected, then time-shifted.

    The shift MUST be applied before the SNR is measured. _linear_time_shift is
    non-circular, so the inspiral pushed past the start of the 8 s segment is
    truncated -- a chirp_mass = 10 system carries ~4.7 s of band from 20 Hz, and
    shifting it by 1 s really does remove signal power. Measuring before the
    shift would leave the realised SNR drifting along the time-shift axis, which
    is precisely the axis the grid controls for.
    """
    signal = injection_generator.signal(theta)
    waveform = signal["waveform"]
    if shift_seconds:
        domain = injection_generator.data_domain
        waveform = {
            det: inf._linear_time_shift(domain, h, float(shift_seconds))
            for det, h in waveform.items()
        }
    return waveform, signal["asds"]


def snr_of(injection_generator, theta, shift_seconds=None):
    waveform, asds = projected_signal(injection_generator, theta, shift_seconds)
    return network_snr(waveform, asds, injection_generator.data_domain.delta_f)


def distance_for_snr(injection_generator, theta, target_snr, shift_seconds=None):
    """Luminosity distance that puts this signal at exactly `target_snr`.

    No search is needed. The polarizations are generated once at a fiducial
    100 Mpc (prior.py:split_off_extrinsic_parameters) and ProjectOntoDetectors
    multiplies them by d_ref / d_new before projecting
    (detector_transforms.py:165), so the detector strain is exactly ~ 1/d and
    SNR(d) = k / d. One waveform evaluation fixes k.

    Returns (distance, snr_at_current_distance).
    """
    snr_now, _ = snr_of(injection_generator, theta, shift_seconds)
    if snr_now <= 0:
        raise ValueError("Signal has zero SNR; cannot rescale to a target SNR.")
    d_now = float(theta["luminosity_distance"])
    return d_now * snr_now / float(target_snr), snr_now


# --------------------------------------------------------------------------
# Separation between the two sources
#
# Luminosity distance is not a radial coordinate you can do vector algebra
# with: at d_L = 2000 Mpc (z = 0.363) the comoving distance is 1467 Mpc, 36%
# smaller. Any 3-D separation has to be built from comoving distances.
#
# NOTE both quantities below are DERIVED: neither enters the likelihood. The
# data is h1 + h2, which responds to each signal's amplitude, arrival time and
# sky position -- not to how far apart the sources are. The sky separation does
# act on the data (through the antenna patterns and the H1-L1 delay); the 3-D
# separation mostly does not, and binning by it mixes a distant intruder in the
# same patch of sky with an equally loud one 60 degrees away.
# --------------------------------------------------------------------------

import functools


@functools.lru_cache(maxsize=1)
def _distance_table():
    """(d_L, comoving) on a dense grid, for interpolation.

    z_at_value() per run would be thousands of root-finds; the relation is
    monotonic, so one interpolation table is exact to well under a Mpc.
    """
    from astropy.cosmology import Planck15

    z = np.concatenate([[0.0], np.geomspace(1e-4, 3.0, 4000)])
    return (Planck15.luminosity_distance(z).value,
            Planck15.comoving_distance(z).value)


def comoving_distance(luminosity_distance):
    d_lum, d_com = _distance_table()
    return np.interp(luminosity_distance, d_lum, d_com)


def sky_separation(ra1, dec1, ra2, dec2):
    """Angle between two sky positions, in degrees."""
    cos_gamma = (np.sin(dec1) * np.sin(dec2)
                 + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2))
    return np.degrees(np.arccos(np.clip(cos_gamma, -1.0, 1.0)))


def place_at_separation(ra1, dec1, gamma, rng):
    """A sky position exactly `gamma` radians from (ra1, dec1).

    The inverse of sky_separation: that one measures the angle between two
    given positions, this one puts the second position at a prescribed angle.
    The azimuth around the axis through signal 1 is uniform, so the result is
    drawn uniformly from the small circle of angular radius gamma -- every
    orientation of the pair is equally likely, and only the separation is
    controlled.

    Returns (ra2, dec2) in radians, ra2 wrapped to [0, 2*pi).
    """
    n1 = np.array([np.cos(dec1) * np.cos(ra1),
                   np.cos(dec1) * np.sin(ra1),
                   np.sin(dec1)])
    # Any vector not parallel to n1; the z axis fails only near the poles.
    helper = np.array([0.0, 0.0, 1.0]) if abs(n1[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(n1, helper)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(n1, e1)

    phi = rng.uniform(0.0, 2.0 * np.pi)
    n2 = (np.cos(gamma) * n1
          + np.sin(gamma) * (np.cos(phi) * e1 + np.sin(phi) * e2))

    dec2 = float(np.arcsin(np.clip(n2[2], -1.0, 1.0)))
    ra2 = float(np.arctan2(n2[1], n2[0]) % (2.0 * np.pi))
    return ra2, dec2


def transverse_separation(d_lum, gamma_deg):
    """D = d_L * gamma, the arc length between two sources at a COMMON distance.

    Distinct from source_separation() above, which is the 3-D chord in comoving
    coordinates. With both sources at the same d_L the chord is 2 d sin(gamma/2)
    and this arc is d gamma; they agree for small gamma and diverge at large.
    """
    return np.asarray(d_lum, dtype=float) * np.radians(gamma_deg)


def source_separation(d1, ra1, dec1, d2, ra2, dec2):
    """Comoving 3-D distance between the two sources, in Mpc.

    d1, d2 are LUMINOSITY distances; they are converted here.
    """
    c1, c2 = comoving_distance(d1), comoving_distance(d2)
    cos_gamma = np.cos(np.radians(sky_separation(ra1, dec1, ra2, dec2)))
    return np.sqrt(np.maximum(c1**2 + c2**2 - 2.0 * c1 * c2 * cos_gamma, 0.0))


# --------------------------------------------------------------------------
# Model / injection-generator construction without a GPU
# --------------------------------------------------------------------------

def metadata_only(main_path):
    """Checkpoint metadata via a lazy mmap, so the diagnostics can build a prior
    and generate waveforms on a login node with no CUDA and no 1.5 GB read."""
    import torch

    ckpt = torch.load(main_path, map_location="cpu", mmap=True, weights_only=False)
    return ckpt["metadata"]


def injection_generator_from_metadata(metadata):
    """inference.build_injection_generator() only ever touches `.metadata`, so a
    stand-in object is enough and the network itself never has to be loaded."""
    return inf.build_injection_generator(types.SimpleNamespace(metadata=metadata))


def resolve_model(model_spec):
    """(label, main_path, init_path) for a MODEL_REGISTRY key or explicit pair."""
    return inf.build_models(model_spec)


def inference_parameters(metadata):
    return list(metadata["train_settings"]["data"]["inference_parameters"])


# --------------------------------------------------------------------------
# Small IO helpers
# --------------------------------------------------------------------------

def write_json(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=float)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def sweep_dir(sweep_name):
    return os.path.join(RESULTS_DIR, sweep_name)


def cell_path(sweep_name, x_index, y_index):
    return os.path.join(sweep_dir(sweep_name), "cells", f"x{x_index:02d}_y{y_index:02d}.hdf5")


def rel(path):
    return os.path.relpath(path, EXAMPLE_DIR)


def short_cell_threshold(n_total, num_realizations, derived):
    """Below how many runs is a cell 'incomplete'?

    A controlled grid has a fixed target per cell, so anything short of it lost
    work. A re-binned grid has no such target: equal-count bins land within a
    few percent of the average (96-103 out of a nominal 100 is normal), and
    log-spaced bins are unequal BY DESIGN. Testing those against the nominal
    count flags most of the grid as broken when nothing is wrong.

    Shared by summarize_grid and plot_heatmap so the console and the figure
    cannot disagree about which cells are trustworthy.
    """
    if not derived:
        return num_realizations
    filled = n_total[n_total > 0]
    return 0.5 * float(np.median(filled)) if filled.size else 0.0
