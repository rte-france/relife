import numpy as np

from .decoder import AdvancedCensoredLifetime, BaseCensoredLifetime, Truncation


# factory
def lifetimes(
    lifetime_values: np.ndarray,
    right_indicators: np.ndarray = np.array([], dtype=bool),
):
    if len(lifetime_values.shape) == 1:
        constructor = BaseCensoredLifetime(lifetime_values)
    elif len(lifetime_values.shape) == 2:
        constructor = AdvancedCensoredLifetime(lifetime_values)
    else:
        return ValueError("lifetimes values must be 1d or 2d array")
    constructor.build(values=lifetime_values, right_indicators=right_indicators)
    return constructor


# factory
def truncations(
    lifetime_values: np.ndarray,
    entry: np.ndarray = np.array([], dtype=float),
    departure: np.ndarray = np.array([], dtype=float),
):
    constructor = Truncation(lifetime_values)
    constructor.build(values=lifetime_values, entry=entry, departure=departure)
    return constructor
