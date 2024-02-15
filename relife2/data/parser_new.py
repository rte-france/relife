from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Parser(ABC):
    """
    Left, right or regular parser
    data pass to parser is unknown but outputs are always 1D array (index) and 1D array (values)
    """

    def __init__(self, *data):
        self.index, self.values = self.parse(*data)
        assert len(self.index.shape) == 1, "index of Parser must be 1d array"
        assert type(self.values) == tuple, "values of Parser must be tuple"
        for _values in self.values:
            assert len(_values.shape) == 1, "values of Parser must be tuple of 1d array"

    @abstractmethod
    def parse(self, *data) -> Tuple[np.ndarray, tuple]:
        pass

    def __call__(self, return_values=True):
        if return_values:
            return self.values
        else:
            return self.index


class CensoredFromIndicators(Parser):
    def __init__(self, censored_lifetimes, indicators):
        assert (
            len(censored_lifetimes.shape) == 1
        ), "data must be 1d array for CensorshipFromIndicators"
        assert len(censored_lifetimes) == len(indicators)
        assert indicators.dtype == bool
        super().__init__(censored_lifetimes, indicators)

    def parse(self, censored_lifetimes, indicators):
        index = np.where(indicators)[0]
        values = (censored_lifetimes[index],)
        return index, values


class ObservedFromIndicators(Parser):
    def __init__(self, censored_lifetimes, indicators):
        assert (
            len(censored_lifetimes.shape) == 1
        ), "data must be 1d array for ObservedFromIndicators"
        assert len(censored_lifetimes) == len(indicators)
        assert indicators.dtype == bool
        super().__init__(censored_lifetimes, indicators)

    def parse(self, censored_lifetimes, indicators):
        # index = (np.stack(indicators)).all(axis=0)
        index = np.where(indicators)[0]
        values = (censored_lifetimes[index],)
        return index, values


class Observed(Parser):
    def __init__(self, censored_lifetimes):
        assert len(censored_lifetimes.shape) == 2, "data must be 2d array for Observed"

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 0] == censored_lifetimes[:, 1])[0]
        values = (censored_lifetimes[index][:, 0],)
        return index, values


class LeftCensored(Parser):
    def __init__(self, censored_lifetimes):
        assert (
            len(censored_lifetimes.shape) == 2
        ), "data must be 2d array for LeftCensorship"
        super().__init__(censored_lifetimes)

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 0] == 0.0)[0]
        values = (censored_lifetimes[index, :][:, 1],)
        return index, values


class RightCensored(Parser):
    def __init__(self, censored_lifetimes):
        assert (
            len(censored_lifetimes.shape) == 2
        ), "data must be 2d array for RightCensorship"
        super().__init__(censored_lifetimes)

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 1] == np.inf)[0]
        values = (censored_lifetimes[index, :][:, 0],)
        return index, values


class IntervalCensored(Parser):
    def __init__(self, censored_lifetimes):
        assert (
            len(censored_lifetimes.shape) == 2
        ), "data must be 2d array for IntervalCensorship"
        super().__init__(censored_lifetimes)

    def parse(self, censored_lifetimes):
        index = np.where(
            np.logical_and(
                np.logical_and(
                    censored_lifetimes[:, 0] > 0,
                    censored_lifetimes[:, 1] < np.inf,
                ),
                np.not_equal(censored_lifetimes[:, 0], censored_lifetimes[:, 1]),
            )
        )[0]
        values = (
            censored_lifetimes[index][:, 0],
            censored_lifetimes[index][:, 1],
        )
        return index, values


class Truncated(Parser):
    def __init__(self, truncation_values):
        assert len(truncation_values.shape) == 1
        super().__init__(truncation_values)

    def parse(self, truncation_values):
        index = np.where(truncation_values > 0)[0]
        values = (truncation_values[index],)
        return index, values


class IntervalTruncated(Parser):
    def __init__(self, left_truncation_values, right_truncation_values):
        assert (
            len(left_truncation_values.shape) == 1
        ), "left_truncation_values must be 1d array for IntervalTruncation"
        assert (
            len(right_truncation_values.shape) == 1
        ), "left_truncation_values must be 1d array for IntervalTruncation"
        assert len(left_truncation_values) == len(right_truncation_values)
        super().__init__(left_truncation_values, right_truncation_values)

    def parse(self, left_truncation_values, right_truncation_values):
        index = np.where(
            np.logical_and(left_truncation_values > 0, right_truncation_values > 0)
        )[0]
        values = (
            left_truncation_values[index],
            right_truncation_values[index],
        )
        return index, values


# factory
def observed_parser(censored_lifetimes: np.ndarray, indicators=None) -> Parser:
    if len(censored_lifetimes.shape) == 1 and indicators is None:
        return ObservedFromIndicators(
            censored_lifetimes, np.ones_like(censored_lifetimes, dtype=bool)
        )
    elif len(censored_lifetimes.shape) == 1 and indicators:
        return ObservedFromIndicators(censored_lifetimes, indicators)
    elif len(censored_lifetimes.shape) == 2 and indicators is None:
        return Observed(censored_lifetimes)
    elif len(censored_lifetimes.shape) == 2 and indicators:
        raise ValueError(
            "observed_parser with 2d censored_lifetimes and indicators is ambiguous"
        )
    else:
        raise ValueError("observed_parser incorrect arguments")


# factory
def left_censored_parser(censored_lifetimes: np.ndarray, indicators=None) -> Parser:
    if len(censored_lifetimes.shape) == 1 and indicators is None:
        return CensoredFromIndicators(
            censored_lifetimes, np.zeros_like(censored_lifetimes, dtype=bool)
        )
    elif len(censored_lifetimes.shape) == 1 and indicators:
        return CensoredFromIndicators(censored_lifetimes, indicators)
    elif len(censored_lifetimes.shape) == 2 and indicators is None:
        return LeftCensored(censored_lifetimes)
    elif len(censored_lifetimes.shape) == 2 and indicators:
        raise ValueError(
            "left_censored_parser with 2d censored_lifetimes and indicators is ambiguous"
        )
    else:
        raise ValueError("left_censored_parser incorrect arguments")


# factory
def right_censored_parser(censored_lifetimes: np.ndarray, indicators=None) -> Parser:
    if len(censored_lifetimes.shape) == 1 and indicators is None:
        return CensoredFromIndicators(
            censored_lifetimes, np.zeros_like(censored_lifetimes, dtype=bool)
        )
    elif len(censored_lifetimes.shape) == 1 and indicators:
        return CensoredFromIndicators(censored_lifetimes, indicators)
    elif len(censored_lifetimes.shape) == 2 and indicators is None:
        return RightCensored(censored_lifetimes)
    elif len(censored_lifetimes.shape) == 2 and indicators:
        raise ValueError(
            "right_censored_parser with 2d censored_lifetimes and indicators is ambiguous"
        )
    else:
        raise ValueError("right_censored_parser incorrect arguments")


# factory
def interval_censored_parser(censored_lifetimes: np.ndarray) -> Parser:
    if len(censored_lifetimes.shape) == 1:
        raise ValueError("interval_censored_parser must take 2d array")
    elif len(censored_lifetimes) == 2:
        return IntervalCensored(censored_lifetimes)
    else:
        raise ValueError("interval_censored_parser incorrect arguments")


# factory
def truncated_parser(
    left_truncation_values=None, right_truncation_values=None
) -> Parser:
    if left_truncation_values and right_truncation_values:
        return IntervalTruncated(left_truncation_values, right_truncation_values)
    elif left_truncation_values and not right_truncation_values:
        return Truncated(left_truncation_values)
    elif not left_truncation_values and right_truncation_values:
        return Truncated(right_truncation_values)
    else:
        return Truncated(np.array([], dtype=float))
