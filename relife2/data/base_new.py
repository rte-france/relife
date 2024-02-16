from dataclasses import InitVar, dataclass
from typing import Union

import numpy as np

from .parser_new import (
    Parser,
    interval_censored_parser_factory,
    left_censored_parser_factory,
    observed_parser_factory,
    right_censored_parser_factory,
    truncated_parser_factory,
)


@dataclass
class CensoredData:
    censored_lifetimes: np.ndarray
    left_censored_indicators: np.ndarray = None
    right_cennsored_indicators: np.ndarray = None
    observed_parser: InitVar[Union[Parser, None]] = None
    left_censored_parser: InitVar[Union[Parser, None]] = None
    right_censored_parser: InitVar[Union[Parser, None]] = None
    interval_censored_parser: InitVar[Union[Parser, None]] = None

    def __post_init__(
        self,
        observed_parser,
        left_censored_parser,
        right_censored_parser,
        interval_censored_parser,
    ):
        if left_censored_parser is None:
            left_censored_parser = left_censored_parser_factory(
                self.censored_lifetimes, indicators=self.left_censored_indicators
            )
        else:
            assert issubclass(left_censored_parser, Parser)
        if right_censored_parser is None:
            right_censored_parser = right_censored_parser_factory(
                self.censored_lifetimes, indicators=self.right_censored_indicators
            )
        else:
            assert issubclass(right_censored_parser, Parser)
        if interval_censored_parser is None:
            interval_censored_parser = interval_censored_parser_factory(
                self.censored_lifetimes
            )
        else:
            assert issubclass(interval_censored_parser, Parser)

        for name in ["values", "index"]:
            setattr(
                self,
                f"left_censored_{name}",
                getattr(left_censored_parser, f"{name}"),
            )
            setattr(
                self,
                f"right_censored_{name}",
                getattr(right_censored_parser, f"{name}"),
            )
            setattr(
                self,
                f"interval_censored_{name}",
                getattr(interval_censored_parser, f"{name}"),
            )

        observed_indicators = None
        if self.left_censored_indicators:
            observed_indicators = np.stack([not self.left_censored_indicators])
            if self.right_cennsored_indicators:
                observed_indicators = np.stack(
                    [observed_indicators, not self.right_cennsored_indicators]
                )
            observed_indicators = observed_indicators.all(axis=0)

        if observed_parser is None:
            observed_parser = observed_parser_factory(
                self.censored_lifetimes, indicators=observed_indicators
            )
        else:
            assert issubclass(observed_parser, Parser)

        for name in ["values", "index"]:
            setattr(
                self,
                f"observed_{name}",
                getattr(observed_parser, f"{name}"),
            )

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


@dataclass
class TruncatedData:
    entry: np.ndarray = None
    departure: np.ndarray = None
    left_truncated_parser: InitVar[Union[Parser, None]] = None
    right_truncated_parser: InitVar[Union[Parser, None]] = None
    interval_truncated_parser: InitVar[Union[Parser, None]] = None

    def __post_init__(
        self, left_truncated_parser, right_truncated_parser, interval_truncated_parser
    ):
        if left_truncated_parser is None:
            left_truncated_parser = truncated_parser_factory(
                left_truncation_values=self.entry, right_truncation_values=None
            )
        else:
            assert issubclass(left_truncated_parser, Parser)
        if right_truncated_parser is None:
            right_truncated_parser = truncated_parser_factory(
                left_truncation_values=None, right_truncation_values=self.departure
            )
        else:
            assert issubclass(right_truncated_parser, Parser)
        if interval_truncated_parser is None:
            interval_truncated_parser = truncated_parser_factory(
                left_truncation_values=self.entry,
                right_truncation_values=self.departure,
            )
        else:
            assert issubclass(interval_truncated_parser, Parser)
        for name in ["values", "index"]:
            setattr(
                self,
                f"left_truncated_{name}",
                getattr(left_truncated_parser, f"{name}"),
            )
            setattr(
                self,
                f"right_truncated_{name}",
                getattr(right_truncated_parser, f"{name}"),
            )
            setattr(
                self,
                f"interval_truncated_{name}",
                getattr(interval_truncated_parser, f"{name}"),
            )

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


@dataclass
class Data:
    censored_data: CensoredData
    truncated_data: TruncatedData

    def __len__(self):
        return len(self.censored_data.censored_lifetimes)

    def observed(self, **kwargs):
        return self.censored_data.observed(kwargs)

    def censored(self, **kwargs):
        return self.censored_data.censored(kwargs)

    def truncated(self, **kwargs):
        return self.truncated_data.truncated(kwargs)

    def info(self):
        headers = ["Lifetime data", "Counts"]
        info = []
        info.append(["Tot.", self.__len__()])
        info.append(["Observed", len(self.censored_data.observed())])
        info.append(["Left censored", len(self.censored_data.censored(how="left"))])
        info.append(["Right censored", len(self.censored_data.censored(how="right"))])
        info.append(
            ["Interval censored", len(self.censored_data.censored(how="interval"))]
        )
        info.append(["Left truncated", len(self.truncated_data.truncated(how="left"))])
        info.append(
            ["Right truncated", len(self.truncated_data.truncated(how="right"))]
        )
        info.append(
            ["Interval truncated", len(self.truncated_data.truncated(how="interval"))]
        )

        row_format = "{:>18}" * (len(headers))
        print(row_format.format(*headers))
        for row in info:
            print(row_format.format(*row))


# # factory
# def survdata(censored_lifetimes: np.ndarray, **kwargs):
#     censored_data = CensoredData(censored_lifetimes, kwargs)
#     truncated_data = TruncatedData(kwargs)
#     # assert on censored vs truncated coherence

#     if np.in1d(
#         censored_data.censored(how="interval"), truncated_data.truncated(how="left")
#     ).any():
#         raise ValueError("interval censored and left truncated lifetimes is ambiguous")
#     if censored_data.observed(return_values=True) < truncated_data.truncated(how="left", return_values=True)

#     return Data(censored_data, truncated_data)
