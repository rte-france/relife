from __future__ import annotations

import copy
from dataclasses import dataclass, field, InitVar
from itertools import product, zip_longest
from typing import Optional, Self
import numpy as np
from numpy._typing import NDArray
from numpy.typing import NDArray


def _time_reshape(time: NDArray[np.float64]) -> NDArray[np.float64]:
    # Check time shape
    if time.ndim > 2 or (time.ndim == 2 and time.shape[-1] not in (1, 2)):
        raise ValueError(f"Invalid time shape, got {time.shape} be time must be (m,), (m, 1) or (m,2)")
    if time.ndim < 2:
        time = time.reshape(-1, 1)  # time is (m, 1) or (m, 2)
    return time


def _event_reshape(event: Optional[NDArray[np.bool_]] = None) -> Optional[NDArray[np.bool_]]:
    if event is not None:
        if event.ndim > 2 or (event.ndim == 2 and event.shape[-1] != 1):
            raise ValueError(f"Invalid event shape, got {event.shape} be event must be (m,) or (m, 1)")
        if event.ndim < 2:
            event = event.reshape(-1, 1)
        return event


def _entry_reshape(entry: Optional[NDArray[np.float64]] = None) -> Optional[NDArray[np.float64]]:
    if entry is not None:
        if entry.ndim > 2 or (entry.ndim == 2 and entry.shape[-1] != 1):
            raise ValueError(f"Invalid entry shape, got {entry.shape} be entry must be (m,) or (m, 1)")
        if entry.ndim < 2:
            entry = entry.reshape(-1, 1)
        return entry


def _departure_reshape(departure: Optional[NDArray[np.float64]] = None) -> Optional[NDArray[np.float64]]:
    if departure is not None:
        if departure.ndim > 2 or (departure.ndim == 2 and departure.shape[-1] != 1):
            raise ValueError(f"Invalid departure shape, got {departure.shape} be departure must be (m,) or (m, 1)")
        if departure.ndim < 2:
            departure = departure.reshape(-1, 1)
        return departure


def _args_reshape(args: tuple[float | NDArray[np.float64], ...] = ()) -> tuple[NDArray[np.float64]]:
    args: list[NDArray[np.float64]] = [np.asarray(arg) for arg in args]
    for i, arg in enumerate(args):
        if arg.ndim > 2:
            raise ValueError(f"Invalid arg shape, got {arg.shape} shape at position {i}")
        if arg.ndim < 2:
            args[i] = arg.reshape(-1, 1)
    return tuple(args)


@dataclass
class LifetimeData:
    time: NDArray[np.float64]
    event: Optional[NDArray[np.bool_]] = field(default=None)
    entry: Optional[NDArray[np.float64]] = field(default=None)
    departure: Optional[NDArray[np.float64]] = field(default=None)
    args: tuple[float | NDArray[np.float64], ...] = field(default_factory=tuple)

    def __post_init__(self):
        self.time = _time_reshape(self.time)
        if self.time.shape[1] == 2 and self.event is not None:
            raise ValueError(
                "When time is 2d, event is not necessary because time already encodes event information. Remove event"
            )
        self.event = _event_reshape(self.event)
        self.entry = _entry_reshape(self.entry)
        self.departure = _departure_reshape(self.departure)
        self.args = _args_reshape(self.args)

        if self.event is None:
            self.event = np.ones((len(self.time), 1)).astype(np.bool_)
        if self.entry is None:
            self.entry = np.zeros((len(self.time), 1))
        if self.departure is None:
            self.departure = np.ones((len(self.time), 1)) * np.inf

        # Check sizes
        sizes = [len(x) for x in (self.time, self.event, self.entry, self.departure, *self.args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {set(sizes)}"
            )


@dataclass
class IndexedLifetimeData:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: NDArray[np.float64]  # (m, 1)
    index: NDArray[np.int64]  # (m, 1)
    args: Optional[tuple[float | NDArray[np.float64], ...]] = field(repr=False, default_factory=tuple)

    def __post_init__(self):
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        if self.values.ndim > 2:
            raise ValueError("IndexData values can't have more than 2 dimensions")
        if len(self.values) != len(self.index):
            raise ValueError("Incompatible lifetime values and index")

    def __len__(self) -> int:
        return len(self.index)

    def union(self, *others: Self) -> Self:
        # return IndexedData(
        #     np.concatenate(
        #         [other.values for other in others],
        #         axis=0,
        #     ),
        #     np.concatenate([other.index for other in others]),
        # )
        other_values = np.concatenate(
            [other.values for other in others],
            axis=0,
        )
        values = np.concatenate([self.values, other_values])
        other_index = np.concatenate([other.index for other in others])
        index = np.concatenate([self.index, other_index])

        other_args = tuple((np.concatenate(x) for x in product(*(other.args for other in others))))
        args = tuple((np.concatenate(x) for x in product(self.args, other_args)))

        sort_ind = np.argsort(
            index
        )  # FIXME: orders of the values seems to affects estimations of the parameters in Regression

        return IndexedLifetimeData(values[sort_ind], index[sort_ind], tuple((arg[index] for arg in args)))


def get_complete(lifetime_data: LifetimeData) -> Optional[IndexedLifetimeData]:
    if lifetime_data.time.shape[-1] == 1:  # 1D time
        index = np.where(lifetime_data.event)[0]
        if index.size > 0:
            values = lifetime_data.time[index]
            args = tuple((arg[index].copy() for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(lifetime_data.time[:, 0] == lifetime_data.time[:, 1])[0]
        if index.size > 0:
            values = lifetime_data.time[index, 0]
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_left_censoring(lifetime_data: LifetimeData) -> Optional[IndexedLifetimeData]:
    if lifetime_data.time.shape[-1] == 1:  # 1D time
        return None
    else:  # 2D time
        index = np.where(lifetime_data.time[:, 0] == 0)[0]
        if index.size > 0:
            values = lifetime_data.time[index, 1]
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_right_censoring(lifetime_data: LifetimeData) -> Optional[IndexedLifetimeData]:
    if lifetime_data.time.shape[-1] == 1:  # 1D time
        index = np.where(~lifetime_data.event)[0]
        if index.size > 0:
            values = lifetime_data.time[index]
            args = tuple((arg[index].copy() for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(lifetime_data.time[:, 1] == np.inf)[0]
        if index.size > 0:
            values = lifetime_data.time[index, 0]
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_interval_censoring(lifetime_data: LifetimeData) -> Optional[IndexedLifetimeData]:
    if lifetime_data.time.shape[-1] == 1:  # 1D time
        rc_index = np.where(~lifetime_data.event)[0]
        if rc_index.size > 0:
            rc_values = np.c_[lifetime_data.time[rc_index], np.ones(len(rc_index)) * np.inf]  # add a column of inf
            args = tuple((arg[rc_index].copy() for arg in lifetime_data.args))
            return IndexedLifetimeData(rc_values, rc_index, args)
        return None
    else:  # 2D time
        index = np.where(
            np.not_equal(lifetime_data.time[:, 0], lifetime_data.time[:, 1]),
        )[0]
        if index.size > 0:
            values = lifetime_data.time[index]
            if values.size != 0:
                if np.any(values[:, 0] >= values[:, 1]):
                    raise ValueError("Interval censorships lower bounds can't be higher or equal to its upper bounds")
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_left_truncation(lifetime_data: LifetimeData) -> Optional[IndexedLifetimeData]:
    if lifetime_data.time.shape[-1] == 1:  # 1D time
        index = np.where(lifetime_data.entry > 0)[0]
        if index.size > 0:
            values = lifetime_data.entry[index]
            args = tuple((arg[index].copy() for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(lifetime_data.entry > 0)[0]
        if index.size > 0:
            values = lifetime_data.entry[index]
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_right_truncation(lifetime_data: LifetimeData) -> Optional[IndexedLifetimeData]:
    if lifetime_data.time.shape[-1] == 1:  # 1D time
        index = np.where(lifetime_data.departure < np.inf)[0]
        if index.size > 0:
            values = lifetime_data.departure[index]
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(lifetime_data.departure < np.inf)[0]
        if index.size > 0:
            values = lifetime_data.departure[index]
            args = tuple((arg[index] for arg in lifetime_data.args))
            return IndexedLifetimeData(values, index, args)
        return None


@dataclass
class StructuredLifetimeData:
    """BLABLABLA"""

    lifetime_data: InitVar[LifetimeData]

    nb_samples: int = field(init=False)

    complete: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 1)
    right_censoring: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 1)
    left_censoring: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 1)
    interval_censoring: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 2)
    left_truncation: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 1)
    right_truncation: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 1)
    complete_or_right_censored: Optional[IndexedLifetimeData] = field(repr=False, init=False)

    def __len__(self):
        return self.nb_samples

    def __post_init__(self, lifetime_data: LifetimeData):

        self.nb_samples = len(lifetime_data)
        self.complete = get_complete(lifetime_data)
        self.right_censoring = (get_right_censoring(lifetime_data),)
        self.left_censoring = (get_left_censoring(lifetime_data),)
        self.interval_censoring = (get_interval_censoring(lifetime_data),)
        self.left_truncation = (get_left_truncation(lifetime_data),)
        self.right_truncation = (get_right_truncation(lifetime_data),)

        # sanity checks that observed lifetimes are inside truncation bounds
        for field_name in [
            "complete",
            "left_censoring",
            "left_censoring",
            "interval_censoring",
        ]:
            data = getattr(self, field_name)
            if data is not None and self.left_truncation is not None:
                inter_ids = (np.intersect1d(data.index, self.left_truncation.index),)
                intersection_values = np.concatenate(
                    (
                        data.values[np.isin(data.index, inter_ids)],
                        self.left_truncation.values[np.isin(self.left_truncation.index, inter_ids)],
                    ),
                    axis=1,
                )
                if len(intersection_values) != 0:
                    if np.any(
                        # take right bound when left bound is 0, otherwise take the min value of the bounds
                        # for none interval lifetimes, min equals the value
                        np.where(
                            intersection_values[:, [0]] == 0,
                            intersection_values[:, [-2]],
                            np.min(intersection_values[:, :-1], axis=1, keepdims=True),
                            # min of all cols but last
                        )
                        < intersection_values[:, [-1]]  # then check if any is under left truncation bound
                    ):
                        raise ValueError("Some lifetimes are under left truncation bounds")
            if data is not None and self.right_truncation is not None:
                inter_ids = np.intersect1d(data.index, self.right_truncation.index)
                intersection_values = np.concatenate(
                    (
                        data.values[np.isin(data.index, inter_ids)],
                        self.right_truncation.values[np.isin(self.right_truncation.index, inter_ids)],
                    ),
                    axis=1,
                )

                if len(intersection_values) != 0:
                    if np.any(
                        # take left bound when right bound is inf, otherwise take the max value of the bounds
                        # for none interval lifetimes, max equals the value
                        np.where(
                            intersection_values[:, [-2]] == np.inf,
                            intersection_values[:, [0]],
                            np.max(intersection_values[:, :-1], axis=1, keepdims=True),
                            # max of all cols but last
                        )
                        > intersection_values[:, [-1]]  # then check if any is above right truncation bound
                    ):
                        raise ValueError("Some lifetimes are above right truncation bounds")

        # compute complete_or_right_censored
        self.complete_or_right_censored = None
        if self.complete is not None and self.right_censoring is not None:
            values = np.concatenate([self.complete.values, self.right_censoring.values], axis=0)
            index = np.concatenate([self.complete.index, self.right_censoring.index], axis=0)
            args = tuple(
                (np.concatenate(x, axis=0) for x in zip_longest(self.complete.args, self.right_censoring.args))
            )
            # FIXME: orders of the values seems to affects estimations of the parameters in Regression
            sort_index = np.argsort(index)

            self.complete_or_right_censored = IndexedLifetimeData(
                values[sort_index],
                index[sort_index],
                tuple((arg[sort_index] for arg in args)),
            )
        elif self.complete is not None:
            self.complete_or_right_censored = copy.deepcopy(self.complete)
        elif self.right_censoring is not None:
            self.complete_or_right_censored = copy.deepcopy(self.right_censoring)
