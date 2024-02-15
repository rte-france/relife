from dataclasses import dataclass

import numpy as np

from .parser_new import (
    interval_censored_parser,
    left_censored_parser,
    observed_parser,
    right_censored_parser,
    truncated_parser,
)


@dataclass
class Data:
    censored_lifetimes: np.ndarray
    left_censored_indicators: np.ndarray = None
    right_cennsored_indicators: np.ndarray = None
    entry: np.ndarray = None
    departure: np.ndarray = None

    def __post_init__(self):

        _left_censored_parser = left_censored_parser(
            self.censored_lifetimes, indicators=self.left_censored_indicators
        )
        _right_censored_parser = right_censored_parser(
            self.censored_lifetimes, indicators=self.right_censored_indicators
        )
        _interval_censored_parser = interval_censored_parser(self.censored_lifetimes)

        for name in ["values", "index"]:
            setattr(
                self,
                f"left_censored_{name}",
                getattr(_left_censored_parser, f"{name}"),
            )
            setattr(
                self,
                f"right_censored_{name}",
                getattr(_right_censored_parser, f"{name}"),
            )
            setattr(
                self,
                f"interval_censored_{name}",
                getattr(_interval_censored_parser, f"{name}"),
            )

        observed_indicators = None
        if self.left_censored_indicators:
            observed_indicators = np.stack([not self.left_censored_indicators])
            if self.right_cennsored_indicators:
                observed_indicators = np.stack(
                    [observed_indicators, not self.right_cennsored_indicators]
                )
            observed_indicators = observed_indicators.all(axis=0)

        _observed_parser = observed_parser(
            self.censored_lifetimes, indicators=observed_indicators
        )

        for name in ["values", "index"]:
            setattr(
                self,
                f"observed_{name}",
                getattr(_observed_parser, f"{name}"),
            )

        _left_truncated_parser = truncated_parser(
            left_truncation_values=self.entry, right_truncation_values=None
        )
        _right_truncated_parser = truncated_parser(
            left_truncation_values=None, right_truncation_values=self.departure
        )
        _interval_truncated_parser = truncated_parser(
            left_truncation_values=self.entry, right_truncation_values=self.departure
        )

        for name in ["values", "index"]:
            setattr(
                self,
                f"left_truncated_{name}",
                getattr(_left_truncated_parser, f"{name}"),
            )
            setattr(
                self,
                f"right_truncated_{name}",
                getattr(_right_truncated_parser, f"{name}"),
            )
            setattr(
                self,
                f"interval_truncated_{name}",
                getattr(_interval_truncated_parser, f"{name}"),
            )

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
