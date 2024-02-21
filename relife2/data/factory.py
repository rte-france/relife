from typing import Type

import numpy as np

from .object import (
    CensoredFromIndicators,
    Data,
    IntervalCensored,
    IntervalData,
    IntervalTruncated,
    LeftCensored,
    Observed,
    ObservedFromIndicators,
    RightCensored,
    Truncated,
)


# factory
def observed_factory(censored_lifetimes: np.ndarray, indicators=None) -> Type[Data]:
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
def left_censored_factory(
    censored_lifetimes: np.ndarray, indicators=None
) -> Type[Data]:
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
def right_censored_factory(
    censored_lifetimes: np.ndarray, indicators=None
) -> Type[Data]:
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
def interval_censored_factory(censored_lifetimes: np.ndarray) -> Type[IntervalData]:
    if len(censored_lifetimes.shape) == 1:
        return IntervalCensored(np.array([[0, 0]], dtype=float))
    elif len(censored_lifetimes.shape) == 2:
        return IntervalCensored(censored_lifetimes)
    else:
        raise ValueError("interval_censored incorrect arguments")


# factory
def left_truncated_factory(left_truncation_values=None) -> Type[Data]:
    if left_truncation_values is not None:
        return Truncated(left_truncation_values)
    else:
        return Truncated(np.array([], dtype=float))


# factory
def right_truncated_factory(right_truncation_values=None) -> Type[Data]:
    if right_truncation_values is not None:
        return Truncated(right_truncation_values)
    else:
        return Truncated(np.array([], dtype=float))


# factory
def interval_truncated_factory(
    left_truncation_values=None, right_truncation_values=None
) -> Type[IntervalData]:
    if left_truncation_values is not None and right_truncation_values is not None:
        return IntervalTruncated(left_truncation_values, right_truncation_values)
    elif left_truncation_values is not None and right_truncation_values is None:
        return IntervalTruncated(np.array([], dtype=float), np.array([], dtype=float))
    elif left_truncation_values is None and right_truncation_values is not None:
        return IntervalTruncated(np.array([], dtype=float), np.array([], dtype=float))
    else:
        return IntervalTruncated(np.array([], dtype=float), np.array([], dtype=float))
