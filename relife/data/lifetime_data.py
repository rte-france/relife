from __future__ import annotations

import copy
from dataclasses import InitVar, dataclass, field
from itertools import zip_longest
from typing import NamedTuple, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray


def time_reshape(time: NDArray[np.float64]) -> NDArray[np.float64]:
    # Check time shape
    if time.ndim > 2 or (time.ndim == 2 and time.shape[-1] not in (1, 2)):
        raise ValueError(f"Invalid time shape, got {time.shape} be time must be (m,), (m, 1) or (m,2)")
    if time.ndim < 2:
        time = time.reshape(-1, 1)  # time is (m, 1) or (m, 2)
    return time


def event_reshape(event: Optional[NDArray[np.bool_]] = None) -> Optional[NDArray[np.bool_]]:
    if event is not None:
        if event.ndim > 2 or (event.ndim == 2 and event.shape[-1] != 1):
            raise ValueError(f"Invalid event shape, got {event.shape} be event must be (m,) or (m, 1)")
        if event.ndim < 2:
            event = event.reshape(-1, 1)
        return event
    return None


def entry_reshape(entry: Optional[NDArray[np.float64]] = None) -> Optional[NDArray[np.float64]]:
    if entry is not None:
        if entry.ndim > 2 or (entry.ndim == 2 and entry.shape[-1] != 1):
            raise ValueError(f"Invalid entry shape, got {entry.shape} be entry must be (m,) or (m, 1)")
        if entry.ndim < 2:
            entry = entry.reshape(-1, 1)
        return entry
    return None


def departure_reshape(departure: Optional[NDArray[np.float64]] = None) -> Optional[NDArray[np.float64]]:
    if departure is not None:
        if departure.ndim > 2 or (departure.ndim == 2 and departure.shape[-1] != 1):
            raise ValueError(f"Invalid departure shape, got {departure.shape} be departure must be (m,) or (m, 1)")
        if departure.ndim < 2:
            departure = departure.reshape(-1, 1)
        return departure
    return None


def args_reshape(args: tuple[float | NDArray[np.float64], ...] = ()) -> tuple[NDArray[np.float64], ...]:
    args_list: list[NDArray[np.float64]] = [np.asarray(arg) for arg in args]
    for i, arg in enumerate(args_list):
        if arg.ndim > 2:
            raise ValueError(f"Invalid arg shape, got {arg.shape} shape at position {i}")
        if arg.ndim < 2:
            args_list[i] = arg.reshape(-1, 1)
    return tuple(args_list)


def get_complete(
    time: NDArray[np.float64], event: Optional[NDArray[np.bool_]] = None, args: tuple[NDArray[np.float64], ...] = ()
) -> Optional[IndexedLifetimeData]:
    if time.shape[-1] == 1:  # 1D time
        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)
        index = np.where(event)[0]
        if index.size > 0:
            values = time[index]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(time[:, 0] == time[:, 1])[0]
        if index.size > 0:
            values = time[index][:, [0]]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_left_censoring(
    time: NDArray[np.float64], args: tuple[NDArray[np.float64], ...] = ()
) -> Optional[IndexedLifetimeData]:
    if time.shape[-1] == 1:  # 1D time
        return None
    else:  # 2D time
        index = np.where(time[:, 0] == 0)[0]
        if index.size > 0:
            values = time[index][:, [1]]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_right_censoring(
    time: NDArray[np.float64], event: Optional[NDArray[np.bool_]] = None, args: tuple[NDArray[np.float64], ...] = ()
) -> Optional[IndexedLifetimeData]:
    if time.shape[-1] == 1:  # 1D time
        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)
        index = np.where(~event)[0]
        if index.size > 0:
            values = time[index]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(time[:, 1] == np.inf)[0]
        if index.size > 0:
            values = time[index][:, [0]]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_interval_censoring(
    time: NDArray[np.float64], event: Optional[NDArray[np.bool_]] = None, args: tuple[NDArray[np.float64], ...] = ()
) -> Optional[IndexedLifetimeData]:
    if time.shape[-1] == 1:  # 1D time
        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)
        rc_index = np.where(~event)[0]
        if rc_index.size > 0:
            rc_values = np.c_[time[rc_index], np.ones(len(rc_index)) * np.inf]  # add a column of inf
            args = tuple((arg[rc_index] for arg in args))
            return IndexedLifetimeData(rc_values, rc_index, args)
        return None
    else:  # 2D time
        index = np.where(
            np.not_equal(time[:, 0], time[:, 1]),
        )[0]
        if index.size > 0:
            values = time[index]  # (m, 2)
            if values.size != 0:
                if np.any(values[:, 0] >= values[:, 1]):
                    raise ValueError("Interval censorships lower bounds can't be higher or equal to its upper bounds")
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_left_truncation(
    time: NDArray[np.float64], entry: Optional[NDArray[np.float64]] = None, args: tuple[NDArray[np.float64], ...] = ()
) -> Optional[IndexedLifetimeData]:
    if entry is None:
        entry = np.zeros((len(time), 1))
    if time.shape[-1] == 1:  # 1D time
        index = np.where(entry > 0)[0]
        if index.size > 0:
            values = entry[index]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(entry > 0)[0]
        if index.size > 0:
            values = entry[index]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None


def get_right_truncation(
    time: NDArray[np.float64],
    departure: Optional[NDArray[np.float64]] = None,
    args: tuple[NDArray[np.float64], ...] = (),
) -> Optional[IndexedLifetimeData]:
    if departure is None:
        departure = np.ones((len(time), 1)) * np.inf
    if time.shape[-1] == 1:  # 1D time
        index = np.where(departure < np.inf)[0]
        if index.size > 0:
            values = departure[index]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None
    else:  # 2D time
        index = np.where(departure < np.inf)[0]
        if index.size > 0:
            values = departure[index]  # (m, 1)
            args = tuple((arg[index] for arg in args))
            return IndexedLifetimeData(values, index, args)
        return None


Args = TypeVarTuple("Args")


class IndexedLifetimeData(NamedTuple):
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    lifetime_values: NDArray[np.float64]  # (m, 1) or (m, 2)
    lifetime_index: NDArray[np.int64]  # (m,)
    args: tuple[NDArray[np.float64], ...] = ()


@dataclass
class LifetimeData:
    nb_samples: int = field(init=False)

    complete: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 1)
    right_censoring: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 1)
    left_censoring: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 1)
    interval_censoring: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 2)
    left_truncation: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 1)
    right_truncation: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 1)
    complete_or_right_censored: Optional[IndexedLifetimeData] = field(init=False, repr=False)  # values shape (m, 1)

    time: InitVar[NDArray[np.float64]]
    event: InitVar[Optional[NDArray[np.bool_]]] = None
    entry: InitVar[Optional[NDArray[np.float64]]] = None
    departure: InitVar[Optional[NDArray[np.float64]]] = None
    args: InitVar[tuple[*Args]] = ()

    def __len__(self):
        return self.nb_samples

    def __post_init__(self, time, event, entry, departure, args):
        time = time_reshape(time)
        event = event_reshape(event)
        if time.shape[1] == 2:
            if event is not None:
                raise ValueError(
                    "When time is 2d, event is not necessary because time already encodes event information. Remove event"
                )
        entry = entry_reshape(entry)
        departure = departure_reshape(departure)
        args = args_reshape(args)

        # Check sizes
        sizes = [len(x) for x in (time, event, entry, departure, *args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {set(sizes)}"
            )

        # TODO : control here
        self.nb_samples = len(time)
        self.complete = get_complete(time, event, args)
        self.right_censoring = get_right_censoring(time, event, args)
        self.left_censoring = get_left_censoring(time, args)
        self.interval_censoring = get_interval_censoring(time, event, args)
        self.left_truncation = get_left_truncation(time, entry, args)
        self.right_truncation = get_right_truncation(time, departure, args)

        # sanity checks that observed lifetimes are inside truncation bounds
        for field_name in [
            "complete",
            "left_censoring",
            "left_censoring",
            "interval_censoring",
        ]:
            data = getattr(self, field_name)
            if data is not None and self.left_truncation is not None:
                inter_ids = np.intersect1d(data.lifetime_index, self.left_truncation.lifetime_index)
                intersection_values = np.concatenate(
                    (
                        data.lifetime_values[np.isin(data.lifetime_index, inter_ids)],
                        self.left_truncation.lifetime_values[np.isin(self.left_truncation.lifetime_index, inter_ids)],
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
                inter_ids = np.intersect1d(data.lifetime_index, self.right_truncation.lifetime_index)
                intersection_values = np.concatenate(
                    (
                        data.lifetime_values[np.isin(data.lifetime_index, inter_ids)],
                        self.right_truncation.lifetime_values[np.isin(self.right_truncation.lifetime_index, inter_ids)],
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
            values = np.concatenate([self.complete.lifetime_values, self.right_censoring.lifetime_values], axis=0)
            index = np.concatenate([self.complete.lifetime_index, self.right_censoring.lifetime_index], axis=0)
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
