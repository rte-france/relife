from dataclasses import InitVar, dataclass
from typing import Union

import numpy as np

from .parser_new import (
    IntervalParser,
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

        if (
            self.left_censored_indicators is not None
            and self.right_censored_indicators is not None
        ):
            assert (
                np.stack(
                    [self.left_censored_indicators, self.right_censored_indicators]
                ).any(axis=0)
                is False
            ), "left_censored_indicators and right_censored_indicators can't true at the same index"

        if interval_censored_parser is None:
            interval_censored_parser = interval_censored_parser_factory(
                self.censored_lifetimes
            )
        else:
            assert issubclass(interval_censored_parser, IntervalParser)

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

        assert (
            self.interval_censored_values[:, 0] < self.interval_censored_values[:, 1]
        ), "invalid interval censorshie values"

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

        assert (
            len(self.entry.shape) == 1
        ), "entry (left truncation values) must be 1d array"
        assert (
            len(self.departure.shape) == 1
        ), "departure (right truncation values) must be 1d array"

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
            assert issubclass(interval_truncated_parser, IntervalParser)
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

        assert (
            self.interval_truncated_values[:, 0] < self.interval_truncated_values[:, 1]
        ), "invalid interval truncation values"

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

    def __post_init__(self):
        """
        censored; truncated; POSSIBILITY
        right; right; True
        left; right; True
        interval; right; True
        right; left; True
        left; left; False
        interval; left; True
        right; interval; True
        left; interval; False
        interval; interval; True
        """
        if self.truncated_data.entry is not None:
            assert len(self.truncated_data.entry) == len(
                self.censored_data.censored_lifetimes
            ), "entry values (left truncation values) must have the same length as censored_lifetimes"
        if self.truncated_data.departure is not None:
            assert len(self.truncated_data.departure) == len(
                self.censored_data.censored_lifetimes
            ), "departure values (right truncation values) must have the same length as censored_lifetimes"

        if self.censored_and_truncated(how=("left", "left")).size != 0:
            raise ValueError(
                "left censored lifetimes can't be left truncated lifetimes too"
            )
        if self.censored_and_truncated(how=("left", "interval")).size != 0:
            raise ValueError(
                "left censored lifetimes can't be interval truncated lifetimes too"
            )

        # checks on values coherence
        censored_values, truncation_values = self.censored_and_truncated(
            how=("right", "left"), return_values=True
        )
        assert (
            censored_values > truncation_values
        ).all(), f"right censored lifetime values can't be lower or equal to left truncation values: incompatible {censored_values} and {truncation_values}"

        censored_values, truncation_values = self.censored_and_truncated(
            how=("interval", "left"), return_values=True
        )
        assert (
            censored_values[:, 0] > truncation_values
        ).all(), f"interval censored lifetime values can't be lower or equal to left truncation values: incompatible {censored_values} and {truncation_values}"
        assert (
            censored_values[:, 1] > truncation_values
        ).all(), f"interval censored lifetime values can't be lower or equal to left truncation values: incompatible {censored_values} and {truncation_values}"

        censored_values, truncation_values = self.censored_and_truncated(
            how=("right", "interval"), return_values=True
        )
        assert (
            censored_values < truncation_values[:, 1]
        ).all(), f"right censored lifetime values can't be higer or equal to interval of truncation: incompatible {censored_values} and {truncation_values}"

        censored_values, truncation_values = self.censored_and_truncated(
            how=("interval", "interval"), return_values=True
        )
        assert (
            censored_values[:, 0] < truncation_values[:, 0]
        ).all(), f"interval censorship can't be outside of truncation interval: incompatible {censored_values} and {truncation_values}"
        assert (
            censored_values[:, 1] > truncation_values[:, 1]
        ).all(), f"interval censorship can't be outside of truncation interval: incompatible {censored_values} and {truncation_values}"

        censored_values, truncation_values = self.censored_and_truncated(
            how=("right", "right"), return_values=True
        )
        assert (
            censored_values < truncation_values
        ).all(), f"right censored lifetime values can't be higher than right truncations: incompatible {censored_values} and {truncation_values}"

        censored_values, truncation_values = self.censored_and_truncated(
            how=("left", "right"), return_values=True
        )
        assert (
            censored_values < truncation_values
        ).all(), f"left censored lifetime values can't be higher than right truncations: incompatible {censored_values} and {truncation_values}"

        censored_values, truncation_values = self.censored_and_truncated(
            how=("interval", "right"), return_values=True
        )
        assert (
            censored_values[:, 1] < truncation_values
        ).all(), f"interval censored lifetime values can't be higher than right truncations: incompatible {censored_values} and {truncation_values}"

        # checks on values coherence
        observed_values, truncation_values = self.observed_and_truncated(
            how="right", return_values=True
        )
        assert (
            observed_values < truncation_values
        ).all(), f"observed lifetime values can't be higher than right truncations: incompatible {observed_values} and {truncation_values}"

        observed_values, truncation_values = self.observed_and_truncated(
            how="left", return_values=True
        )
        assert (
            observed_values > truncation_values
        ).all(), f"observed lifetime values can't be lower than left truncations: incompatible {observed_values} and {truncation_values}"

        observed_values, truncation_values = self.observed_and_truncated(
            how="interval", return_values=True
        )
        assert (
            observed_values < truncation_values[:, 1]
        ).all(), f"observed lifetime values can't be outside of truncation interval: incompatible {observed_values} and {truncation_values}"
        assert (
            observed_values > truncation_values[:, 0]
        ).all(), f"observed lifetime values can't be outside of truncation interval: incompatible {observed_values} and {truncation_values}"

    def __len__(self):
        return len(self.censored_data.censored_lifetimes)

    def observed(self, **kwargs):
        """return observed lifetime index or values

        Returns:
            _type_: _description_
        """
        return self.censored_data.observed(kwargs)

    def censored(self, **kwargs):
        """return censored lifetime index or censorship values

        Returns:
            _type_: _description_
        """
        return self.censored_data.censored(kwargs)

    def truncated(self, **kwargs):
        """return truncated lifetime index or truncation values

        Returns:
            _type_: _description_
        """
        return self.truncated_data.truncated(kwargs)

    def censored_and_truncated(self, how=("right", "left"), return_values=False):
        """return censored and truncated (simultaneously) lifetimes index or corresponding censorship values and truncation values

        Args:
            how (tuple, optional): _description_. Defaults to ("right", "left").
            return_values (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        truncated_index = self.truncated(how=how[1])
        censored_index = self.censored(how=how[0])
        index_censored_and_truncated = truncated_index[
            np.in1d(truncated_index, censored_index)
        ]

        if return_values:
            censored_lifetime_values = self.censored(how=how[0], return_values=True)
            truncation_values = self.truncated(how=how[1], return_values=True)
            res_1 = censored_lifetime_values[np.in1d(censored_index, truncated_index)]
            res_2 = truncation_values[np.in1d(truncated_index, censored_index)]
            return res_1, res_2

        else:
            index_censored_and_truncated

    def observed_and_truncated(self, how="left", return_values=False):
        """return observed and truncated (simultaneously) lifetimes index or corresponding observed values and truncation values

        Args:
            how (str, optional): _description_. Defaults to "left".
            return_values (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        truncated_index = self.truncated(how=how)
        observed_index = self.observed()
        index_observed_and_truncated = observed_index[
            np.in1d(observed_index, truncated_index)
        ]

        if return_values:
            observed_lifetime_values = self.observed(return_values=True)
            truncation_values = self.truncated(how=how, return_values=True)
            res_1 = observed_lifetime_values[np.in1d(observed_index, truncated_index)]
            res_2 = truncation_values[np.in1d(truncated_index, observed_index)]
            return res_1, res_2
        else:
            return index_observed_and_truncated

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


# factory
def survdata(censored_lifetimes: np.ndarray, **kwargs):
    censored_data = CensoredData(censored_lifetimes, kwargs)
    truncated_data = TruncatedData(kwargs)
    return Data(censored_data, truncated_data)
