from typing import Tuple

import numpy as np

from ..errors import LifetimeDataError
from ..errors.checks import (
    check_shape,
    is_1d,
    is_array,
    is_array_type,
    same_shape,
)


class ParsedData:
    """
    data pass to parser is unknown but outputs are always 1D array (index) and 1D array (values)
    """

    def __init__(self, *data: np.ndarray):
        self.index, self.values = self.parse(*data)

    def parse(self, *data) -> Tuple[np.ndarray]:
        return data

    def __repr__(self):
        class_name = type(self).__name__
        return (
            f"{class_name}(index={repr(self.index)},"
            f" values={repr(self.values)})"
        )

    def __len__(self):
        return len(self.index)


class CensoredFromIndicators(ParsedData):
    def __init__(self, observed_lifetimes: np.ndarray, indicators: np.ndarray):
        super().__init__(observed_lifetimes, indicators)

    def parse(self, observed_lifetimes, indicators):
        index = np.where(indicators)[0]
        values = observed_lifetimes[index]
        return index, values


class CompleteFromIndicators(ParsedData):
    def __init__(self, observed_lifetimes: np.ndarray, indicators: np.ndarray):
        super().__init__(observed_lifetimes, indicators)

    def parse(self, observed_lifetimes, indicators):
        # index = (np.stack(indicators)).all(axis=0)
        index = np.where(indicators)[0]
        values = observed_lifetimes[index]
        return index, values


class CompleteFromIntervals(ParsedData):
    def __init__(self, observed_lifetimes: np.ndarray):
        super().__init__(observed_lifetimes)

    def parse(self, observed_lifetimes):
        index = np.where(observed_lifetimes[:, 0] == observed_lifetimes[:, 1])[
            0
        ]
        values = observed_lifetimes[index][:, 0]
        return index, values


class LCFromIntervals(ParsedData):
    def __init__(self, observed_lifetimes: np.ndarray):
        super().__init__(observed_lifetimes)

    def parse(self, observed_lifetimes):
        index = np.where(observed_lifetimes[:, 0] == 0.0)[0]
        values = observed_lifetimes[index, :][:, 1]
        return index, values


class RCFromIntervals(ParsedData):
    def __init__(self, observed_lifetimes: np.ndarray):
        super().__init__(observed_lifetimes)

    def parse(self, observed_lifetimes):
        index = np.where(observed_lifetimes[:, 1] == np.inf)[0]
        values = observed_lifetimes[index, :][:, 0]
        return index, values


class ICFromIntervals(ParsedData):
    def __init__(self, observed_lifetimes: np.ndarray):
        super().__init__(observed_lifetimes)

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


class LeftTruncated(ParsedData):
    def __init__(
        self,
        left_truncation_values: np.ndarray,
        right_truncation_values: np.ndarray,
    ):
        super().__init__(left_truncation_values, right_truncation_values)

    def parse(
        self,
        left_truncation_values: np.ndarray,
        right_truncation_values: np.ndarray,
    ):
        index = np.where(
            np.logical_and(
                left_truncation_values > 0, right_truncation_values == 0
            )
        )[0]
        values = left_truncation_values[index]
        return index, values


class RightTruncated(ParsedData):
    def __init__(
        self,
        left_truncation_values: np.ndarray,
        right_truncation_values: np.ndarray,
    ):
        super().__init__(left_truncation_values, right_truncation_values)

    def parse(
        self,
        left_truncation_values: np.ndarray,
        right_truncation_values: np.ndarray,
    ):
        index = np.where(
            np.logical_and(
                right_truncation_values > 0, left_truncation_values == 0
            )
        )[0]
        values = right_truncation_values[index]
        return index, values


class IntervalTruncated(ParsedData):
    def __init__(
        self,
        left_truncation_values: np.ndarray,
        right_truncation_values: np.ndarray,
    ):
        super().__init__(left_truncation_values, right_truncation_values)

    def parse(
        self,
        left_truncation_values: np.ndarray,
        right_truncation_values: np.ndarray,
    ):
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


# factory
def complete_factory(
    observed_lifetimes: np.ndarray, indicators: np.ndarray = None
) -> CompleteFromIndicators:

    if observed_lifetimes.ndim == 1 and indicators is None:
        return CompleteFromIndicators(
            observed_lifetimes, np.ones_like(observed_lifetimes, dtype=bool)
        )

    elif observed_lifetimes.ndim == 1 and indicators is not None:

        try:
            is_array(indicators)
            is_array_type(indicators, np.bool_)
            same_shape(observed_lifetimes, indicators)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error
        return CompleteFromIndicators(observed_lifetimes, indicators)

    elif observed_lifetimes.ndim == 2 and indicators is None:

        try:
            check_shape(observed_lifetimes.shape, 2, 1)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error
        return CompleteFromIntervals(observed_lifetimes)

    elif observed_lifetimes.ndim == 2 and indicators is not None:
        raise LifetimeDataError(
            "Complete lifetimes from 2d observed_lifetimes with indicators are"
            " ambiguous"
        )
    else:
        raise LifetimeDataError("observed_lifetimes must be 1d or 2d np.array")


# factory
def left_censored_factory(
    observed_lifetimes: np.ndarray, indicators: np.ndarray = None
) -> LCFromIntervals:

    if observed_lifetimes.ndim == 1 and indicators is None:
        return CensoredFromIndicators(
            observed_lifetimes, np.zeros_like(observed_lifetimes, dtype=bool)
        )

    elif observed_lifetimes.ndim == 1 and indicators is not None:

        try:
            is_array(indicators)
            is_array_type(indicators, np.bool_)
            same_shape(observed_lifetimes, indicators)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return CensoredFromIndicators(observed_lifetimes, indicators)

    elif observed_lifetimes.ndim == 2 and indicators is None:

        try:
            check_shape(observed_lifetimes.shape, 2, 1)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return LCFromIntervals(observed_lifetimes)
    elif observed_lifetimes.ndim == 2 and indicators is not None:
        raise LifetimeDataError(
            "Left censored lifetimes from 2d observed_lifetimes with"
            " indicators are ambiguous"
        )
    else:
        raise LifetimeDataError("observed_lifetimes must be 1d or 2d np.array")


# factory
def right_censored_factory(
    observed_lifetimes: np.ndarray, indicators: np.ndarray = None
) -> RCFromIntervals:

    if observed_lifetimes.ndim == 1 and indicators is None:
        return CensoredFromIndicators(
            observed_lifetimes, np.zeros_like(observed_lifetimes, dtype=bool)
        )

    elif observed_lifetimes.ndim == 1 and indicators is not None:
        try:
            is_array(indicators)
            is_array_type(indicators, np.bool_)
            same_shape(observed_lifetimes, indicators)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error
        return CensoredFromIndicators(observed_lifetimes, indicators)

    elif observed_lifetimes.ndim == 2 and indicators is None:
        try:
            check_shape(observed_lifetimes.shape, 2, 1)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error
        return RCFromIntervals(observed_lifetimes)

    elif observed_lifetimes.ndim == 2 and indicators is not None:

        raise LifetimeDataError(
            "Right censored lifetimes from 2d observed_lifetimes with"
            " indicators are ambiguous"
        )
    else:
        raise LifetimeDataError("observed_lifetimes must be 1d or 2d np.array")


# factory
def interval_censored_factory(
    observed_lifetimes: np.ndarray,
) -> ICFromIntervals:

    if observed_lifetimes.ndim == 1:
        return ICFromIntervals(np.array([[0, 0]], dtype=float))
    elif observed_lifetimes.ndim == 2:
        try:
            check_shape(observed_lifetimes.shape, 2, 1)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error
        return ICFromIntervals(observed_lifetimes)
    else:
        raise LifetimeDataError("observed_lifetimes must be 1d or 2d np.array")


# factory
def left_truncated_factory(
    left_truncation_values: np.ndarray = None,
    right_truncation_values: np.ndarray = None,
) -> LeftTruncated:

    if left_truncation_values is not None and right_truncation_values is None:

        try:
            is_array(left_truncation_values)
            is_1d(left_truncation_values)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return LeftTruncated(
            left_truncation_values, np.zeros_like(left_truncation_values)
        )

    elif (
        left_truncation_values is not None
        and right_truncation_values is not None
    ):

        try:
            is_array(left_truncation_values)
            is_array(right_truncation_values)
            is_1d(left_truncation_values)
            same_shape(left_truncation_values, right_truncation_values)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return LeftTruncated(left_truncation_values, right_truncation_values)
    else:
        return LeftTruncated(
            np.array([], dtype=float), np.array([], dtype=float)
        )


# factory
def right_truncated_factory(
    left_truncation_values: np.ndarray = None,
    right_truncation_values: np.ndarray = None,
) -> RightTruncated:
    if right_truncation_values is not None and left_truncation_values is None:

        try:
            is_array(right_truncation_values)
            is_1d(right_truncation_values)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return RightTruncated(
            np.zeros_like(left_truncation_values), right_truncation_values
        )

    elif (
        left_truncation_values is not None
        and right_truncation_values is not None
    ):

        try:
            is_array(left_truncation_values)
            is_array(right_truncation_values)
            is_1d(left_truncation_values)
            same_shape(left_truncation_values, right_truncation_values)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return RightTruncated(left_truncation_values, right_truncation_values)

    else:
        return RightTruncated(
            np.array([], dtype=float), np.array([], dtype=float)
        )


# factory
def interval_truncated_factory(
    left_truncation_values: np.ndarray = None,
    right_truncation_values: np.ndarray = None,
) -> IntervalTruncated:
    if (
        left_truncation_values is not None
        and right_truncation_values is not None
    ):

        try:
            is_array(left_truncation_values)
            is_array(right_truncation_values)
            is_1d(left_truncation_values)
            same_shape(left_truncation_values, right_truncation_values)
        except Exception as error:
            raise LifetimeDataError("Invalid lifetime data") from error

        return IntervalTruncated(
            left_truncation_values, right_truncation_values
        )
    else:
        return IntervalTruncated(
            np.array([], dtype=float), np.array([], dtype=float)
        )
