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

    def __repr__(self):
        class_name = type(self).__name__
        return (
            f"{class_name}(index={repr(self.index)},"
            f" values={repr(self.values)})"
        )

    def __len__(self):
        return len(self.index)


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

    def __repr__(self):
        class_name = type(self).__name__
        return (
            f"{class_name}(index={repr(self.index)},"
            f" values={repr(self.values)})"
        )

    def __len__(self):
        return len(self.index)


class ExtractedData:
    def __init__(self, index, values):
        if type(index) != np.ndarray:
            raise TypeError("Expected np.ndarray index")
        if type(values) != np.ndarray:
            raise TypeError("Expected np.ndarray values")
        if len(index.shape) != 1:
            raise TypeError("index must be of shape (n,)")
        if len(values.shape) != 1 and len(values.shape) != 2:
            raise TypeError("values must be of shape (n,) or (n, 2)")
        if len(values.shape) == 2:
            if values.shape[-1] != 2:
                raise TypeError("values must be of shape (n,) or (n, 2)")
        if len(index) != len(values):
            raise ValueError("index and values must have the same length")
        self.index = index
        self.values = values

    def __len__(self):
        return len(self.index)


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


class CompleteObservationsFromIndicators(Data):
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


class CompleteObservations(Data):
    def __init__(self, censored_lifetimes):
        super().__init__(censored_lifetimes)
        if len(censored_lifetimes.shape) != 2:
            raise TypeError("data must be 2d array")

    def parse(self, censored_lifetimes):
        index = np.where(censored_lifetimes[:, 0] == censored_lifetimes[:, 1])[
            0
        ]
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
                np.not_equal(
                    censored_lifetimes[:, 0], censored_lifetimes[:, 1]
                ),
            )
        )[0]
        values = censored_lifetimes[index]
        return index, values


class LeftTruncated(Data):
    def __init__(self, left_truncation_values, right_truncation_values):
        super().__init__(left_truncation_values, right_truncation_values)
        if len(left_truncation_values.shape) != 1:
            raise TypeError("left truncation values must be 1d array")
        if len(right_truncation_values.shape) != 1:
            raise TypeError("right truncation values must be 1d array")
        if len(left_truncation_values) != len(right_truncation_values):
            raise ValueError(
                "left_truncation_values and right_truncation_values must have"
                " the same length"
            )

    def parse(self, left_truncation_values, right_truncation_values):
        index = np.where(
            np.logical_and(
                left_truncation_values > 0, right_truncation_values == 0
            )
        )[0]
        values = left_truncation_values[index]
        return index, values


class RightTruncated(Data):
    def __init__(self, left_truncation_values, right_truncation_values):
        super().__init__(left_truncation_values, right_truncation_values)
        if len(left_truncation_values.shape) != 1:
            raise TypeError("truncation values must be 1d array")
        if len(right_truncation_values.shape) != 1:
            raise TypeError("truncation values must be 1d array")

    def parse(self, left_truncation_values, right_truncation_values):
        index = np.where(
            np.logical_and(
                right_truncation_values > 0, left_truncation_values == 0
            )
        )[0]
        values = right_truncation_values[index]
        return index, values


class IntervalTruncated(IntervalData):
    def __init__(self, left_truncation_values, right_truncation_values):
        super().__init__(left_truncation_values, right_truncation_values)
        if len(left_truncation_values.shape) != 1:
            raise TypeError(
                "left_truncation_values must be 1d array for"
                " IntervalTruncation"
            )
        if len(right_truncation_values.shape) != 1:
            raise TypeError(
                "left_truncation_values must be 1d array for"
                " IntervalTruncation"
            )
        if len(left_truncation_values) != len(right_truncation_values):
            raise ValueError(
                "left_truncation_values and right_truncation_values must have"
                " the same length"
            )

    def parse(self, left_truncation_values, right_truncation_values):
        index = np.where(
            np.logical_and(
                left_truncation_values > 0, right_truncation_values > 0
            )
        )[0]
        values = np.concatenate(
            (
                left_truncation_values[index][:, None],
                right_truncation_values[index][:, None],
            ),
            axis=1,
        )
        return index, values
