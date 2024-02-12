from typing import Union

import numpy as np

from ._resources import AdvancedCensoredLifetime, BaseCensoredLifetime, Truncation


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


class SurvivalData:
    def __init__(
        self,
        lifetime_values: Union[list, np.ndarray],
        event: np.ndarray = np.array([], dtype=bool),
        entry: np.ndarray = np.array([], dtype=float),
        covar: np.ndarray = np.array([[]], dtype=float),
    ):
        self.lifetime_values = lifetime_values
        self.lifetime_data = lifetimes(lifetime_values, right_indicators=1 - event)
        self.truncation_data = truncations(lifetime_values, entry=entry)
        self.covar = covar

    @property
    def lifetimes(self):
        return self.lifetime_values

    def __len__(self):
        return len(self.lifetime_values)

    @property
    def observed_values(self):
        return self.lifetime_data.regular_values

    @property
    def right_censored_values(self):
        return self.lifetime_data.right_values

    @property
    def left_censored_values(self):
        return self.lifetime_data.left_values

    @property
    def interval_censored_values(self):
        return self.lifetime_data.interval_values

    @property
    def observed_index(self):
        return self.lifetime_data.regular_index

    @property
    def right_censored_index(self):
        return self.lifetime_data.right_index

    @property
    def left_censored_index(self):
        return self.lifetime_data.left_index

    @property
    def interval_censored_index(self):
        return self.lifetime_data.interval_index

    @property
    def left_truncations(self):
        return self.truncation_data.left_values

    @property
    def left_truncated_index(self):
        return self.truncation_data.left_index

    @property
    def right_truncations(self):
        return self.truncation_data.right_values

    @property
    def right_truncated_index(self):
        return self.truncation_data.right_index

    @property
    def interval_truncations(self):
        return self.truncation_data.interval_values

    @property
    def interval_truncated_index(self):
        return self.truncation_data.interval_index

    def observed(self, return_values: bool = False):
        if return_values:
            return getattr(self, "observed_values")
        else:
            return getattr(self, "observed_index")

    def censored(self, how: str = "right", return_values: bool = False):
        assert how in [
            "right",
            "left",
            "interval",
        ], f"how must be left, right or interval. Not {how}"
        if return_values:
            return getattr(self, f"{how}_censored_values")
        else:
            return getattr(self, f"{how}_censored_index")

    def truncated(self, how: str = "right", return_values: bool = False):
        assert how in [
            "right",
            "left",
            "interval",
        ], f"how must be left, right or interval. Not {how}"
        if return_values:
            return getattr(self, f"{how}_truncations")
        else:
            return getattr(self, f"{how}_truncated_index")

    def info(self):
        headers = ["Lifetime data", "Counts"]
        info = []
        info.append(["Tot.", self.__len__()])
        info.append(["Observed", len(self.observed())])
        info.append(["Left censored", len(self.censored(how="left"))])
        info.append(["Right censored", len(self.censored(how="right"))])
        info.append(["Interval censored", len(self.censored(how="interval"))])
        info.append(["Left truncated", len(self.truncated(how="left"))])
        info.append(["Right truncated", len(self.truncated(how="right"))])
        info.append(["Interval truncated", len(self.truncated(how="interval"))])

        row_format = "{:>18}" * (len(headers))
        print(row_format.format(*headers))
        for row in info:
            print(row_format.format(*row))
