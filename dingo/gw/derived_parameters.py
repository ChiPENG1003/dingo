"""
Parameters that are deterministic functions of the waveform parameters.

Dingo's flow learns the joint density over whatever list of names appears in
``data.inference_parameters``. Those names do not have to be waveform-generation
parameters: any quantity that is a fixed function of the parameters stored in the
waveform dataset can be added to the list. The waveform is still generated from the
full parameter set, so listing a derived parameter (and dropping the parameters it is
built from) makes the network model the *marginal* posterior of that quantity,
marginalized over everything else under the training prior.

The trade-off is that such a network no longer provides a full parameter vector, so
its samples cannot be used for importance sampling / likelihood evaluation.

To add a new derived parameter, register a function below. It receives a mapping of
the standard dingo parameters (a DataFrame, or a dict of arrays or floats) and returns
the derived values.
"""

from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd

# Note: dingo's `mass_ratio` follows the bilby convention q = mass_2 / mass_1 <= 1.


def _chi_eff(p):
    """Effective aligned spin, (m_1 a_1 cos(tilt_1) + m_2 a_2 cos(tilt_2)) / (m_1 + m_2)."""
    q = p["mass_ratio"]
    return (
        p["a_1"] * np.cos(p["tilt_1"]) + q * p["a_2"] * np.cos(p["tilt_2"])
    ) / (1 + q)


def _chi_p(p):
    """Effective precession spin (Schmidt et al., arXiv:1408.1810)."""
    q = p["mass_ratio"]
    return np.maximum(
        p["a_1"] * np.sin(p["tilt_1"]),
        (4 * q + 3) / (3 * q + 4) * q * p["a_2"] * np.sin(p["tilt_2"]),
    )


def _chi_1z(p):
    return p["a_1"] * np.cos(p["tilt_1"])


def _chi_2z(p):
    return p["a_2"] * np.cos(p["tilt_2"])


_SPIN_DEPENDENCIES = ["mass_ratio", "a_1", "a_2", "tilt_1", "tilt_2"]

# name -> (function, names of the parameters it is computed from)
DERIVED_PARAMETERS = {
    "chi_eff": (_chi_eff, _SPIN_DEPENDENCIES),
    "chi_p": (_chi_p, _SPIN_DEPENDENCIES),
    "chi_1z": (_chi_1z, ["a_1", "tilt_1"]),
    "chi_2z": (_chi_2z, ["a_2", "tilt_2"]),
}


def select_derived_parameters(
    names: Iterable[str], available: Optional[Iterable[str]] = None
) -> list:
    """
    Return the subset of `names` that are registered derived parameters.

    Parameters
    ----------
    names : Iterable[str]
        Candidate parameter names (e.g., inference_parameters).
    available : Iterable[str], optional
        Names that are already present as ordinary parameters. These are skipped,
        so a parameter stored in the dataset is never recomputed.

    Returns
    -------
    list[str]
        Derived parameter names, in the order they appear in `names`, deduplicated.
    """
    available = set(available) if available is not None else set()
    selected = []
    for name in names:
        if name in DERIVED_PARAMETERS and name not in available and name not in selected:
            selected.append(name)
    return selected


def check_derived_parameters(names: Iterable[str]):
    """Raise if any of `names` is not a registered derived parameter."""
    unknown = [n for n in names if n not in DERIVED_PARAMETERS]
    if unknown:
        raise ValueError(
            f"Unknown derived parameter(s) {unknown}. Registered derived parameters "
            f"are {sorted(DERIVED_PARAMETERS)}. Add new ones in "
            f"dingo/gw/derived_parameters.py."
        )


def _compute(name: str, parameters: Mapping):
    fn, dependencies = DERIVED_PARAMETERS[name]
    missing = [d for d in dependencies if d not in parameters]
    if missing:
        raise KeyError(
            f"Cannot compute derived parameter '{name}': the parameters {missing} are "
            f"not available. It requires {dependencies}."
        )
    return fn(parameters)


def add_derived_parameters(parameters: pd.DataFrame, names: Iterable[str]):
    """
    Add derived parameters as new columns of a parameter DataFrame, in place.

    Parameters
    ----------
    parameters : pd.DataFrame
        Parameters of a waveform dataset. Modified in place.
    names : Iterable[str]
        Registered derived parameters to add.
    """
    check_derived_parameters(names)
    for name in names:
        parameters[name] = np.asarray(
            _compute(name, parameters), dtype=parameters.dtypes.iloc[0]
        )


def with_derived_parameters(
    parameters: Mapping, names: Optional[Iterable[str]] = None
) -> dict:
    """
    Return a copy of a parameter dict, extended with derived parameters.

    Parameters
    ----------
    parameters : Mapping
        Parameters (e.g., an injection's truth values). Not modified.
    names : Iterable[str], optional
        Derived parameters to add. By default, every registered derived parameter
        whose dependencies are present and which is not already in `parameters`.

    Returns
    -------
    dict
    """
    if names is None:
        names = [
            name
            for name, (_, dependencies) in DERIVED_PARAMETERS.items()
            if name not in parameters and all(d in parameters for d in dependencies)
        ]
    else:
        check_derived_parameters(names)

    out = dict(parameters)
    for name in names:
        value = _compute(name, parameters)
        out[name] = float(value) if np.ndim(value) == 0 else value
    return out
