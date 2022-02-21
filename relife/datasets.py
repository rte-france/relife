"""Lifetime datasets to load."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import numpy as np
from pathlib import Path

from .data import LifetimeData


DATA_PATH = Path(__file__).parent / "datasets"


def load_power_transformer() -> LifetimeData:
    """Load and return the power transformer lifetime data.

    The data were simulated from a real estimate:

    - `time`: time-to-event or durations in years,
    - `event`: if a failure occurs during the observation period,
    - `entry`: age of the power transformers in years at the beginning of the
      observation period.

    Returns
    -------
    LifetimeData
        The lifetime data as a dataclass instance.

    Examples
    --------

    .. code-block::

        from relife.datasets import load_power_transformer
        time, event, entry = load_power_transformer().astuple()
    """
    data = np.loadtxt(
        DATA_PATH / "power_transformer.csv", delimiter=",", skiprows=1, unpack=True
    )
    return LifetimeData(*data)


def load_circuit_breaker() -> LifetimeData:
    """Load and return the circuit breaker lifetime data.

    The data were simulated from a real estimate:

    - `time`: time-to-event or durations in years,
    - `event`: if a failure occurs during the observation period,
    - `entry`: age of the circuit breakers in years at the beginning of the
      observation period.

    Returns
    -------
    LifetimeData
        The lifetime data as a dataclass instance.

    Examples
    --------

    .. code-block::

        from relife.datasets import load_circuit_breaker
        time, event, entry = load_circuit_breaker().astuple()
    """
    data = np.loadtxt(
        DATA_PATH / "circuit_breaker.csv", delimiter=",", skiprows=1, unpack=True
    )
    return LifetimeData(*data)


def load_insulator_string() -> LifetimeData:
    """Load and return the insulator string lifetime data for regression.

    The data were simulated from a real estimate:

    - `time`: time-to-event or durations in years,
    - `event`: if a failure occurs during the observation period,
    - `entry`: age of the circuit breakers in years at the beginning of the
      observation period,
    - `args`: tuple of covariates related to the atmospheric polluants.

    Returns
    -------
    LifetimeData
        The lifetime data as a dataclass instance.

    Examples
    --------

    .. code-block::

        import numpy as np
        from scipy.stats import boxcox, zscore
        from relife.datasets import load_circuit_breaker
        time, event, entry, *args = load_insulator_string().astuple()
        covar = zscore(np.column_stack([boxcox(col)[0] for col in args[0].T]))
    """
    time, event, entry, *args = np.loadtxt(
        DATA_PATH / "insulator_string.csv", delimiter=",", skiprows=1, unpack=True
    )
    covar = np.column_stack(args)
    return LifetimeData(time, event, entry, (covar,))
