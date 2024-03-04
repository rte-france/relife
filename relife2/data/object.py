from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from ..utils import (
    has_same_length,
    is_1d,
    is_1d_or_2d,
    is_2d,
    is_array,
    is_array_type,
)


class Data(ABC):
    """
    data pass to parser is unknown but outputs are always 1D array (index) and 1D array (values)
    """

    def __init__(self, *data):
        try:
            is_array(*data)
        except Exception as error:
            raise ValueError("invalid argument") from error
        self.index, self.values = self.parse(*data)
        try:
            is_1d(self.index)
            is_1d(self.values)
            has_same_length(self.index, self.values)
        except Exception as error:
            raise ValueError("invalid argument") from error

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
        try:
            is_array(*data)
        except Exception as error:
            raise ValueError("invalid argument") from error
        self.index, self.values = self.parse(*data)
        try:
            is_1d(self.index)
            is_2d(self.values)
            has_same_length(self.index, self.values)
        except Exception as error:
            raise ValueError("invalid argument") from error

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
        try:
            is_array(index)
            is_array(values)
            is_1d(index)
            is_1d_or_2d(values)
            has_same_length(index, values)
        except Exception as error:
            raise ValueError("invalid argument") from error
        self.index = index
        self.values = values

    def __len__(self):
        return len(self.index)


class CensoredFromIndicators(Data):
    def __init__(self, observed_lifetimes, indicators):
        super().__init__(observed_lifetimes, indicators)
        try:
            is_1d(observed_lifetimes)
            is_1d(indicators)
            is_array_type(indicators, np.bool_)
            has_same_length(observed_lifetimes, indicators)
        except Exception as error:
            raise ValueError("invalid argument") from error

    def parse(self, observed_lifetimes, indicators):
        index = np.where(indicators)[0]
        values = observed_lifetimes[index]
        return index, values


class CompleteObservationsFromIndicators(Data):
    def __init__(self, observed_lifetimes, indicators):
        super().__init__(observed_lifetimes, indicators)
        try:
            is_1d(observed_lifetimes)
            is_1d(indicators)
            is_array_type(indicators, np.bool_)
            has_same_length(observed_lifetimes, indicators)
        except Exception as error:
            raise ValueError("invalid argument") from error

    def parse(self, observed_lifetimes, indicators):
        # index = (np.stack(indicators)).all(axis=0)
        index = np.where(indicators)[0]
        values = observed_lifetimes[index]
        return index, values


class CompleteObservations(Data):
    def __init__(self, observed_lifetimes):
        super().__init__(observed_lifetimes)
        try:
            is_2d(observed_lifetimes)
        except Exception as error:
            raise ValueError("invalid argument") from error

    def parse(self, observed_lifetimes):
        index = np.where(observed_lifetimes[:, 0] == observed_lifetimes[:, 1])[
            0
        ]
        values = observed_lifetimes[index][:, 0]
        return index, values


class LeftCensored(Data):
    def __init__(self, observed_lifetimes):
        super().__init__(observed_lifetimes)
        try:
            is_2d(observed_lifetimes)
        except Exception as error:
            raise ValueError("invalid argument") from error

    def parse(self, observed_lifetimes):
        index = np.where(observed_lifetimes[:, 0] == 0.0)[0]
        values = observed_lifetimes[index, :][:, 1]
        return index, values


class RightCensored(Data):
    def __init__(self, observed_lifetimes):
        super().__init__(observed_lifetimes)
        try:
            is_2d(observed_lifetimes)
        except Exception as error:
            raise ValueError("invalid argument") from error

    def parse(self, observed_lifetimes):
        index = np.where(observed_lifetimes[:, 1] == np.inf)[0]
        values = observed_lifetimes[index, :][:, 0]
        return index, values


class IntervalCensored(IntervalData):
    def __init__(self, observed_lifetimes):
        super().__init__(observed_lifetimes)
        try:
            is_2d(observed_lifetimes)
        except Exception as error:
            raise ValueError("invalid argument") from error

    def parse(self, observed_lifetimes):
        index = np.where(
            np.logical_and(
                np.logical_and(
                    observed_lifetimes[:, 0] > 0,
                    observed_lifetimes[:, 1] < np.inf,
                ),
                np.not_equal(
                    observed_lifetimes[:, 0], observed_lifetimes[:, 1]
                ),
            )
        )[0]
        values = observed_lifetimes[index]
        return index, values


class LeftTruncated(Data):
    def __init__(self, left_truncation_values, right_truncation_values):
        super().__init__(left_truncation_values, right_truncation_values)
        try:
            is_1d(left_truncation_values)
            is_1d(right_truncation_values)
            has_same_length(left_truncation_values, right_truncation_values)
        except Exception as error:
            raise ValueError("invalid argument") from error

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
        try:
            is_1d(left_truncation_values)
            is_1d(right_truncation_values)
            has_same_length(left_truncation_values, right_truncation_values)
        except Exception as error:
            raise ValueError("invalid argument") from error

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
        try:
            is_1d(left_truncation_values)
            is_1d(right_truncation_values)
            has_same_length(left_truncation_values, right_truncation_values)
        except Exception as error:
            raise ValueError("invalid argument") from error

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
