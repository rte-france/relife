from dataclasses import dataclass
from typing import Union

import numpy as np

from .decoder import censoredlifetimes_decoder, truncations_decoder


@dataclass
class SurvivalData:
    lifetimes: Union[list, np.ndarray]
    event: np.ndarray = np.array([], dtype=bool)
    entry: np.ndarray = np.array([], dtype=float)
    covar: np.ndarray = np.array([[]], dtype=float)

    def __post_init__(self):
        _censoredlifetimes_decoder = censoredlifetimes_decoder(
            self.lifetimes, 1 - self.event
        )
        _truncations_decoder = truncations_decoder(self.lifetimes, self.entry)

        for how in ["left", "right", "interval"]:
            for name in ["values", "index"]:
                setattr(
                    self,
                    f"{how}_censored_{name}",
                    getattr(_censoredlifetimes_decoder, f"get_{how}_{name}")(),
                )
        for how in ["regular"]:
            for name in ["values", "index"]:
                setattr(
                    self,
                    f"{how}_{name}",
                    getattr(_censoredlifetimes_decoder, f"get_{how}_{name}")(),
                )
        for how in ["left", "right", "interval"]:
            for name in ["values", "index"]:
                setattr(
                    self,
                    f"{how}_truncated_{name}",
                    getattr(_truncations_decoder, f"get_{how}_{name}")(),
                )

    def __len__(self):
        return len(self.lifetimes)

    def observed(self, return_values: bool = False):
        if return_values:
            return getattr(self, "regular_values")
        else:
            return getattr(self, "regular_index")

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
            return getattr(self, f"{how}_truncated_values")
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
