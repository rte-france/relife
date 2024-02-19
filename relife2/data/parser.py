from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class Data(ABC):
    """
    data pass to parser is unknown but outputs are always 1D array (index) and 1D array (values)
    """

    def __init__(self, *data):
        if {type(_data) for _data in data} != {np.ndarray}:
            raise TypeError("Expected np.ndarray data")
        self.index, self.values = self.parse(*data)
        if len(self.index.shape) != 1:
            raise TypeError("index of Data must be of shape (n,)")
        if len(self.values.shape) != 1:
            raise TypeError("values of Data must be of shape (n,)")

    @abstractmethod
    def parse(self, *data) -> Tuple[np.ndarray, np.ndarray]:
        pass


class IntervalData(ABC):
    """
    data pass to parser is unknown but outputs are always 1D array (index) and 2D array (values)
    """

    def __init__(self, *data):
        if {type(_data) for _data in data} != {np.ndarray}:
            raise TypeError("Expected np.ndarray data")
        self.index, self.values = self.parse(*data)
        if len(self.index.shape) != 1:
            raise TypeError("index of IntervalParser must be of shape (n,)")
        if len(self.values.shape) != 2:
            raise TypeError("values of IntervalParser must of shape (n, 2)")
        if self.values.shape[-1] != 2:
            raise TypeError("values of IntervalParser must of shape (n, 2)")

    @abstractmethod
    def parse(self, *data) -> Tuple[np.ndarray, np.ndarray]:
        pass


class CensoredFromIndicators(Data):
    def __init__(self, censored_lifetimes, indicators):
        super().__init__(censored_lifetimes, indicators)
        if len(censored_lifetimes.shape) != 1:
            raise TypeError("truncation values must be 1d array")
        if len(indicators.shape) != 1:
            raise TypeError("indicators values must be 1d array")
        if indicators.dtype != bool:
            raise TypeError("indicators values must be boolean")
        if len(censored_lifetimes) != len(indicators):
            raise ValueError(
                "censored_lifetimes and indicators must have the same length"
            )

    def parse(self, censored_lifetimes, indicators):
        index = np.where(indicators)[0]
        values = censored_lifetimes[index]
        return index, values


class ObservedFromIndicators(Data):
    def __init__(self, censored_lifetimes, indicators):
        super().__init__(censored_lifetimes, indicators)
        if len(censored_lifetimes.shape) != 1:
            raise TypeError("truncation values must be 1d array")
        if len(indicators.shape) != 1:
            raise TypeError("indicators values must be 1d array")
        if indicators.dtype != bool:
            raise TypeError("indicators values must be boolean")
        if len(censored_lifetimes) != len(indicators):
            raise ValueError(
                "censored_lifetimes and indicators must have the same length"
            )

    def parse(self, censored_lifetimes, indicators):
        # index = (np.stack(indicators)).all(axis=0)
        index = np.where(indicators)[0]
        values = censored_lifetimes[index]
        return index, values


class Observed(Data):
    def __init__(self, censored_lifetimes):
        super().__init__(censored_lifetimes)
        if len(censored_lifetimes.shape) != 2:
            raise TypeError("data must be 2d array")

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 0] == censored_lifetimes[:, 1])[0]
        values = censored_lifetimes[index][:, 0]
        return index, values


class LeftCensored(Data):
    def __init__(self, censored_lifetimes):
        super().__init__(censored_lifetimes)
        if len(censored_lifetimes.shape) != 2:
            raise TypeError("data must be 2d array")

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 0] == 0.0)[0]
        values = censored_lifetimes[index, :][:, 1]
        return index, values


class RightCensored(Data):
    def __init__(self, censored_lifetimes):
        super().__init__(censored_lifetimes)
        if len(censored_lifetimes.shape) != 2:
            raise TypeError("data must be 2d array")

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 1] == np.inf)[0]
        values = censored_lifetimes[index, :][:, 0]
        return index, values


class IntervalCensored(IntervalData):
    def __init__(self, censored_lifetimes):
        super().__init__(censored_lifetimes)
        if len(censored_lifetimes.shape) != 2:
            raise TypeError("data must be 2d array")
        if len(censored_lifetimes.shape) != 2:
            raise TypeError("expected data of shape (n, 2)")
        if censored_lifetimes.shape[-1] != 2:
            raise TypeError("expected data of shape (n, 2)")

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
        values = censored_lifetimes[index]
        return index, values


class Truncated(Data):
    def __init__(self, truncation_values):
        super().__init__(truncation_values)
        if len(truncation_values.shape) != 1:
            raise TypeError("truncation values must be 1d array")

    def parse(self, truncation_values):
        index = np.where(truncation_values > 0)[0]
        values = truncation_values[index]
        return index, values


class IntervalTruncated(IntervalData):
    def __init__(self, left_truncation_values, right_truncation_values):
        super().__init__(left_truncation_values, right_truncation_values)
        if len(left_truncation_values.shape) != 1:
            raise TypeError(
                "left_truncation_values must be 1d array for IntervalTruncation"
            )
        if len(right_truncation_values.shape) != 1:
            raise TypeError(
                "left_truncation_values must be 1d array for IntervalTruncation"
            )
        if len(left_truncation_values) != len(right_truncation_values):
            raise ValueError(
                "left_truncation_values and right_truncation_values must have the same length"
            )

    def parse(self, left_truncation_values, right_truncation_values):
        index = np.where(
            np.logical_and(left_truncation_values > 0, right_truncation_values > 0)
        )[0]
        values = np.concatenate(
            (
                left_truncation_values[index][:, None],
                right_truncation_values[index][:, None],
            ),
            axis=1,
        )
        return index, values


# factory
def observed_factory(censored_lifetimes: np.ndarray, indicators=None) -> Data:
    if len(censored_lifetimes.shape) == 1 and indicators is None:
        return ObservedFromIndicators(
            censored_lifetimes, np.ones_like(censored_lifetimes, dtype=bool)
        )
    elif len(censored_lifetimes.shape) == 1 and indicators is not None:
        return ObservedFromIndicators(censored_lifetimes, indicators)
    elif len(censored_lifetimes.shape) == 2 and indicators is None:
        return Observed(censored_lifetimes)
    elif len(censored_lifetimes.shape) == 2 and indicators is not None:
        raise ValueError(
            "observed with 2d censored_lifetimes and indicators is ambiguous"
        )
    else:
        raise ValueError("observed_parser incorrect arguments")


# factory
def left_censored_factory(censored_lifetimes: np.ndarray, indicators=None) -> Data:
    if len(censored_lifetimes.shape) == 1 and indicators is None:
        return CensoredFromIndicators(
            censored_lifetimes, np.zeros_like(censored_lifetimes, dtype=bool)
        )
    elif len(censored_lifetimes.shape) == 1 and indicators is not None:
        return CensoredFromIndicators(censored_lifetimes, indicators)
    elif len(censored_lifetimes.shape) == 2 and indicators is None:
        return LeftCensored(censored_lifetimes)
    elif len(censored_lifetimes.shape) == 2 and indicators is not None:
        raise ValueError(
            "left_censored with 2d censored_lifetimes and indicators is ambiguous"
        )
    else:
        raise ValueError("left_censored_parser incorrect arguments")


# factory
def right_censored_factory(censored_lifetimes: np.ndarray, indicators=None) -> Data:
    if len(censored_lifetimes.shape) == 1 and indicators is None:
        return CensoredFromIndicators(
            censored_lifetimes, np.zeros_like(censored_lifetimes, dtype=bool)
        )
    elif len(censored_lifetimes.shape) == 1 and indicators is not None:
        return CensoredFromIndicators(censored_lifetimes, indicators)
    elif len(censored_lifetimes.shape) == 2 and indicators is None:
        return RightCensored(censored_lifetimes)
    elif len(censored_lifetimes.shape) == 2 and indicators is not None:
        raise ValueError(
            "right_censored with 2d censored_lifetimes and indicators is ambiguous"
        )
    else:
        raise ValueError("right_censored_parser incorrect arguments")


# factory
def interval_censored_factory(censored_lifetimes: np.ndarray) -> Data:
    if len(censored_lifetimes.shape) == 1:
        return IntervalCensored(np.array([[0, 0]], dtype=float))
    elif len(censored_lifetimes.shape) == 2:
        return IntervalCensored(censored_lifetimes)
    else:
        raise ValueError("interval_censored incorrect arguments")


# factory
def left_truncated_factory(left_truncation_values=None):
    if left_truncation_values is not None:
        return Truncated(left_truncation_values)
    else:
        return Truncated(np.array([], dtype=float))


# factory
def right_truncated_factory(right_truncation_values=None):
    if right_truncation_values is not None:
        return Truncated(right_truncation_values)
    else:
        return Truncated(np.array([], dtype=float))


# factory
def interval_truncated_factory(
    left_truncation_values=None, right_truncation_values=None
):
    if left_truncation_values is not None and right_truncation_values is not None:
        return IntervalTruncated(left_truncation_values, right_truncation_values)
    elif left_truncation_values is not None and right_truncation_values is None:
        return IntervalTruncated(np.array([], dtype=float), np.array([], dtype=float))
    elif left_truncation_values is None and right_truncation_values is not None:
        return IntervalTruncated(np.array([], dtype=float), np.array([], dtype=float))
    else:
        return IntervalTruncated(np.array([], dtype=float), np.array([], dtype=float))
