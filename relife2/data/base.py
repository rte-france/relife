from typing import Union

import numpy as np

from .factory import lifetimes, truncations


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

    def observed(self, return_values: bool = False):
        if return_values:
            return getattr(self.lifetime_data, "regular_values")
        else:
            return getattr(self.lifetime_data, "regular_index")

    def censored(self, how: str = "right", return_values: bool = False):
        assert how in [
            "right",
            "left",
            "interval",
        ], f"how must be left, right or interval. Not {how}"
        if return_values:
            return getattr(self.lifetime_data, f"{how}_values")
        else:
            return getattr(self.lifetime_data, f"{how}_index")

    def truncated(self, how: str = "right", return_values: bool = False):
        assert how in [
            "right",
            "left",
            "interval",
        ], f"how must be left, right or interval. Not {how}"
        if return_values:
            return getattr(self.truncation_data, f"{how}_values")
        else:
            return getattr(self.truncation_data, f"{how}_index")

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
