"""
This module defines dataclass used to encapsulate data used for parameters estimation of models

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields
from itertools import filterfalse
from typing import Iterator, Optional, Protocol, Self, Sequence

import numpy as np
from numpy.typing import NDArray

from relife.utils.types import ModelArgs


@dataclass
class IndexedData:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: NDArray[np.float64]
    index: NDArray[np.int64]

    def __post_init__(self):
        if self.values.ndim == 1:
            self.values = self.values.reshape(-1, 1)
        if self.values.ndim > 2:
            raise ValueError("IndexData values can't have more than 2 dimensions")
        if len(self.values) != len(self.index):
            raise ValueError("Incompatible lifetime values and index")

    def __len__(self) -> int:
        return len(self.index)

    def intersection(*others: Self) -> Self:
        """
        Args:
            *: LifetimeData object.s containing values of shape (n1, p1), (n2, p2), etc.

        Returns:

        Examples:
            >>> data_1 = IndexedData(values = np.array([[1.], [2.]]), index = np.array([3, 10]))
            >>> data_2 = IndexedData(values = np.array([[3.], [5.]]), index = np.array([10, 2]))
            >>> data_1.intersection(data_2)
            IndexedData(values=array([[2, 3]]), index=array([10]))
        """

        inter_ids = np.array(
            list(set.intersection(*[set(other.index) for other in others]))
        ).astype(np.int64)
        return IndexedData(
            np.concatenate(
                [other.values[np.isin(other.index, inter_ids)] for other in others],
                axis=1,
            ),
            inter_ids,
        )

    def union(*others: Self) -> Self:
        # return IndexedData(
        #     np.concatenate(
        #         [other.values for other in others],
        #         axis=0,
        #     ),
        #     np.concatenate([other.index for other in others]),
        # )
        values = np.concatenate(
            [other.values for other in others],
            axis=0,
        )
        index = np.concatenate([other.index for other in others])
        sort_ind = np.argsort(
            index
        )  # FIXME: orders of the values seems to affects estimations of the parameters in Regression
        return IndexedData(values[sort_ind], index[sort_ind])


@dataclass
class LifetimeData:
    """BLABLABLA"""

    nb_samples: int
    complete: IndexedData = field(repr=False)  # values shape (m, 1)
    left_censoring: IndexedData = field(repr=False)  # values shape (m, 1)
    right_censoring: IndexedData = field(repr=False)  # values shape (m, 1)
    interval_censoring: IndexedData = field(repr=False)  # values shape (m, 2)
    left_truncation: IndexedData = field(repr=False)  # values shape (m, 1)
    right_truncation: IndexedData = field(repr=False)  # values shape (m, 1)

    def __len__(self):
        return self.nb_samples

    def __post_init__(self):
        self.rc = self.right_censoring.union(self.complete)
        self.rc = self.complete.union(self.left_censoring, self.right_censoring)

        # sanity check that observed lifetimes are inside truncation bounds
        for field_name in [
            "complete",
            "left_censoring",
            "left_censoring",
            "interval_censoring",
        ]:
            data = getattr(self, field_name)
            if len(self.left_truncation) != 0 and len(data) != 0:
                intersect_data = data.intersection(self.left_truncation)
                if len(intersect_data) != 0:
                    if np.any(
                        # take right bound when left bound is 0, otherwise take the min value of the bounds
                        # for none interval lifetimes, min equals the value
                        np.where(
                            intersect_data.values[:, [0]] == 0,
                            intersect_data.values[:, [-2]],
                            np.min(
                                intersect_data.values[:, :-1], axis=1, keepdims=True
                            ),
                            # min of all cols but last
                        )
                        < intersect_data.values[
                            :, [-1]
                        ]  # then check if any is under left truncation bound
                    ):
                        raise ValueError(
                            "Some lifetimes are under left truncation bounds"
                        )
            if len(self.right_truncation) != 0 and len(data) != 0:
                intersect_data = data.intersection(self.right_truncation)
                if len(intersect_data) != 0:
                    if np.any(
                        # take left bound when right bound is inf, otherwise take the max value of the bounds
                        # for none interval lifetimes, max equals the value
                        np.where(
                            intersect_data.values[:, [-2]] == np.inf,
                            intersect_data.values[:, [0]],
                            np.max(
                                intersect_data.values[:, :-1], axis=1, keepdims=True
                            ),
                            # max of all cols but last
                        )
                        > intersect_data.values[
                            :, [-1]
                        ]  # then check if any is above right truncation bound
                    ):
                        raise ValueError(
                            "Some lifetimes are above right truncation bounds"
                        )


@dataclass
class Deteriorations:
    """BLABLABLA"""

    values: np.ndarray  # R0 in first column (always)
    times: np.ndarray  # 0 in first column (always)
    ids: np.ndarray

    def __post_init__(self):
        # self.values = np.ma.array(self.values, mask=np.isnan(self.values))
        # self.times = np.ma.array(self.times, mask=np.isnan(self.times))
        self.increments = -np.diff(self.values, axis=1)
        self.event = self.increments == 0


class LifetimeParser(Protocol):
    """
    Factory method of ObservedLifetimes and Truncations
    """

    def __init__(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ):

        if entry is None:
            entry = np.zeros((len(time), 1))

        if departure is None:
            departure = np.ones((len(time), 1)) * np.inf

        if event is None:
            event = np.ones((len(time), 1)).astype(np.bool_)

        self.time = time
        self.event: NDArray[np.bool_] = event.astype(np.bool_)
        self.entry: NDArray[np.float64] = entry
        self.departure: NDArray[np.float64] = departure

    @abstractmethod
    def get_complete(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censoring(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censoring(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censoring(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncation(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncation(self) -> IndexedData:
        """
        Returns:
            IndexedData: object containing right truncations values and index
        """


class Lifetime1DParser(LifetimeParser):
    """
    Concrete implementation of LifetimeDataReader for 1D encoding
    """

    def get_complete(self) -> IndexedData:
        index = np.where(self.event)[0]
        values = self.time[index]
        return IndexedData(values, index)

    def get_left_censoring(self) -> IndexedData:
        return IndexedData(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    def get_right_censoring(self) -> IndexedData:
        index = np.where(~self.event)[0]
        values = self.time[index]
        return IndexedData(values, index)

    def get_interval_censoring(self) -> IndexedData:
        rc_index = np.where(~self.event)[0]
        rc_values = np.c_[
            self.time[rc_index], np.ones(len(rc_index)) * np.inf
        ]  # add a column of inf
        return IndexedData(rc_values, rc_index)

    def get_left_truncation(self) -> IndexedData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return IndexedData(values, index)

    def get_right_truncation(self) -> IndexedData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return IndexedData(values, index)


class Lifetime2DParser(LifetimeParser):
    """
    Concrete implementation of LifetimeDataReader for 2D encoding
    """

    def get_complete(self) -> IndexedData:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0]
        return IndexedData(values, index)

    def get_left_censoring(
        self,
    ) -> IndexedData:
        index = np.where(self.time[:, 0] == 0)[0]
        values = self.time[index, 1]
        return IndexedData(values, index)

    def get_right_censoring(
        self,
    ) -> IndexedData:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0]
        return IndexedData(values, index)

    def get_interval_censoring(self) -> IndexedData:
        index = np.where(
            np.not_equal(self.time[:, 0], self.time[:, 1]),
        )[0]
        values = self.time[index]
        if values.size != 0:
            if np.any(values[:, 0] >= values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return IndexedData(values, index)

    def get_left_truncation(self) -> IndexedData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return IndexedData(values, index)

    def get_right_truncation(self) -> IndexedData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return IndexedData(values, index)


def lifetime_data_factory(
    time: NDArray[np.float64],
    event: Optional[NDArray[np.bool_]] = None,
    entry: Optional[NDArray[np.float64]] = None,
    departure: Optional[NDArray[np.float64]] = None,
) -> LifetimeData:
    """
    Args:
        time ():
        event ():
        entry ():
        departure ():

    Returns:

    """
    reader: LifetimeParser
    if time.ndim == 1:
        reader = Lifetime1DParser(time, event, entry, departure)
    elif time.ndim == 2:
        if time.shape[-1] != 2:
            raise ValueError("If time ndim is 2, time shape must be (n, 2)")
        reader = Lifetime2DParser(time, event, entry, departure)
    else:
        raise ValueError("time ndim must be 1 or 2")

    return LifetimeData(
        len(time),
        reader.get_complete(),
        reader.get_left_censoring(),
        reader.get_right_censoring(),
        reader.get_interval_censoring(),
        reader.get_left_truncation(),
        reader.get_right_truncation(),
    )


def deteriorations_factory(
    deterioration_measurements: np.ndarray,
    inspection_times: np.ndarray,
    unit_ids: np.ndarray,
):
    """
    Args:
        deterioration_measurements ():
        inspection_times ():
        unit_ids ():
    Returns:

    """
    # verifier la cohérence des arguments (temps croissant et mesures decroissantes)

    sorted_indices = np.argsort(unit_ids, kind="mergesort")
    sorted_unit_ids = unit_ids[sorted_indices]
    sorted_deteriorations_measurements = deterioration_measurements[sorted_indices]
    sorted_inspection_times = inspection_times[sorted_indices]
    unique_unit_ids, counts = np.unique(sorted_unit_ids, return_counts=True)

    max_len = np.max(counts)
    split_indices = np.cumsum(counts)[:-1]

    deteriorations_measurements_2d = np.vstack(
        [
            np.concatenate((split_arr, np.ones(max_len - len(split_arr)) * np.nan))
            for split_arr in np.split(sorted_deteriorations_measurements, split_indices)
        ]
    )

    inspection_times_2d = np.vstack(
        [
            np.concatenate((split_arr, np.ones(max_len - len(split_arr)) * np.nan))
            for split_arr in np.split(sorted_inspection_times, split_indices)
        ]
    )

    return Deteriorations(
        deteriorations_measurements_2d,
        inspection_times_2d,
        unique_unit_ids,
    )


@dataclass
class CountData(ABC):
    samples_index: NDArray[np.int64] = field(repr=False)  # samples index
    assets_index: NDArray[np.int64] = field(repr=False)  # assets index
    order: NDArray[np.int64] = field(
        repr=False
    )  # order index (order in generation process)
    event_times: NDArray[np.float64] = field(repr=False)

    nb_samples: int = field(init=False)
    nb_assets: int = field(init=False)
    samples_unique_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique samples index
    assets_unique_index: NDArray[np.int64] = field(
        init=False, repr=False
    )  # unique assets index

    def __post_init__(self):
        fields_values = [
            getattr(self, _field.name) for _field in fields(self) if _field.init
        ]
        if not all(
            arr.ndim == 1 for arr in fields_values if isinstance(arr, np.ndarray)
        ):
            raise ValueError("All array values must be 1d")
        if (
            not len(
                set(
                    arr.shape[0] for arr in fields_values if isinstance(arr, np.ndarray)
                )
            )
            == 1
        ):
            raise ValueError("All array values must have the same shape")

        self.samples_unique_index = np.unique(
            self.samples_index
        )  # samples unique index
        self.assets_unique_index = np.unique(self.assets_index)  # assets unique index
        self.nb_samples = len(self.samples_unique_index)
        self.nb_assets = len(self.assets_unique_index)

    def __len__(self) -> int:
        return self.nb_samples * self.nb_assets

    def number_of_events(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples_index == sample
        times = np.insert(np.sort(self.event_times[ind]), 0, 0)
        counts = np.arange(times.size)
        return times, counts

    def mean_number_of_events(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        times = np.insert(np.sort(self.event_times), 0, 0)
        counts = np.arange(times.size) / self.nb_samples
        return times, counts

    @abstractmethod
    def iter(self, sample: Optional[int] = None) -> "CountDataIterable": ...


class CountDataIterable:
    def __init__(self, data: CountData, field_names: Sequence[str]):
        """
        Parameters
        ----------
        data :
        field_names : fields iterate on
        """
        self.data = data
        sorted_index = np.lexsort((data.order, data.assets_index, data.samples_index))
        self.sorted_fields = tuple(
            (getattr(data, name)[sorted_index].copy() for name in field_names)
        )
        self.samples_index = data.samples_index[sorted_index].copy()
        self.assets_index = data.assets_index[sorted_index].copy()

    def __len__(self) -> int:
        return self.data.nb_samples * self.data.nb_assets

    def __iter__(self) -> Iterator[tuple[int, int, *tuple[NDArray[np.float64], ...]]]:

        for sample in self.data.samples_unique_index:
            sample_mask = self.samples_index == sample
            for asset in self.data.assets_unique_index:
                asset_mask = self.assets_index[sample_mask] == asset
                itervalues = tuple(
                    (v[sample_mask][asset_mask]) for v in self.sorted_fields
                )
                yield int(sample), int(asset), *itervalues


@dataclass
class RenewalData(CountData):
    lifetimes: NDArray[np.float64] = field(repr=False)
    events: NDArray[np.bool_] = field(
        repr=False
    )  # event indicators (right censored or not)

    model_args: ModelArgs = field(repr=False)
    with_model1: bool = field(repr=False)

    # necessary to store reference to model_args used to sample (to_lifetime_data)
    def __post_init__(self):
        super().__post_init__()
        # tile args in 2d (homogeneous shape if multiple assets)
        args_2d = [np.atleast_2d(arg) for arg in self.model_args]
        if self.nb_assets > 1 and bool(self.model_args):
            for i, arg in enumerate(args_2d):
                if arg.shape[0] == 1:
                    args_2d[i] = np.tile(arg, (self.nb_assets, 1))
        self.model_args = tuple(args_2d)

    def iter(self, sample: Optional[int] = None):
        if sample is None:
            return CountDataIterable(self, ("event_times", "lifetimes", "events"))
        else:
            if sample not in self.samples_unique_index:
                raise ValueError(f"{sample} is not part of samples index")
            return filterfalse(
                lambda x: x[0] != sample,
                CountDataIterable(self, ("event_times", "lifetimes", "events")),
            )

    # def _preprocess_args(self) -> tuple[NDArray[np.float64], ...]:
    #     # tile args in 2d (homogeneous shape if multiple assets)
    #     args_2d = [np.atleast_2d(arg) for arg in self.model_args]
    #     if self.nb_assets > 1 and bool(self.model_args):
    #         for i, arg in enumerate(args_2d):
    #             if arg.shape[0] == 1:
    #                 args_2d[i] = np.tile(arg, (self.nb_assets, 1))
    #     return tuple(args_2d)

    def _get_args(
        self, index, previous_args: Optional[tuple[NDArray[np.float64], ...]] = None
    ) -> tuple[NDArray[np.float64], ...]:
        args = ()
        if self.nb_assets > 1 and bool(self.model_args):
            if previous_args:
                args = tuple(
                    (
                        np.concatenate((p, np.take(a, index, axis=0)))
                        for p, a in zip(previous_args, self.model_args)
                    )
                )
            else:
                args = tuple((np.take(a, index, axis=0) for a in self.model_args))
        return args

    def to_fit(
        self,
        t0: float = 0,
        tf: Optional[float] = None,
        sample: Optional[int] = None,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.float64],
        None,
        tuple[NDArray[np.float64], ...],
    ]:
        """
        consider only model_args (not model1_args)
        if t0 is lower than first event_times => raise  Error

        Parameters
        ----------
        t0 : start (time) of the observation period
        tf : end (time) of the observation period
        sample :

        Returns
        -------

        """

        max_event_time = np.max(self.event_times)
        if tf > max_event_time:
            tf = max_event_time

        if t0 >= tf:
            raise ValueError("`t0` must be strictly lower than `tf`")

        time = np.array([], dtype=np.float64)
        event = np.array([], dtype=np.bool_)
        entry = np.array([], dtype=np.float64)

        s = self.samples_index == sample if sample is not None else Ellipsis

        complete_left_truncated = (self.event_times[s] > t0) & (
            self.event_times[s] <= tf
        )

        _timeline = self.event_times[s][complete_left_truncated]
        _lifetimes = self.lifetimes[s][complete_left_truncated]
        _events = self.events[s][complete_left_truncated]
        _assets_index = self.assets_index[s][complete_left_truncated]

        shift_left = _timeline - _lifetimes
        left_truncated = (t0 - shift_left) >= 0
        left_truncations = (t0 - shift_left)[left_truncated]

        time = np.concatenate(
            (time, _lifetimes[left_truncated], _lifetimes[~left_truncated])
        )
        event = np.concatenate(
            (
                event,
                np.ones_like(left_truncations, dtype=np.bool_),
                _events[~left_truncated],
            )
        )
        entry = np.concatenate(
            (entry, left_truncations, np.zeros_like(_lifetimes[~left_truncated]))
        )

        args = self._get_args(_assets_index[left_truncated])
        args = self._get_args(_assets_index[~left_truncated], previous_args=args)
        print(args)

        right_censored = self.event_times[s] > tf

        _timeline = self.event_times[s][right_censored]
        _lifetimes = self.lifetimes[s][right_censored]
        _events = self.events[s][right_censored]
        _assets_index = self.assets_index[s][right_censored]

        shift_left = _timeline - _lifetimes
        right_censored = (tf - shift_left) >= 0
        right_censoring = (tf - shift_left)[right_censored]

        time = np.concatenate((time, right_censoring))
        event = np.concatenate((event, np.zeros_like(right_censoring, dtype=np.bool_)))
        entry = np.concatenate((entry, np.zeros_like(right_censoring)))

        args = self._get_args(_assets_index[right_censored], previous_args=args)

        return time, event, entry, None, args


@dataclass
class RenewalRewardData(RenewalData):
    total_rewards: NDArray[np.float64] = field(repr=False)

    def cum_total_rewards(
        self, sample: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        ind = self.samples_index == sample
        s = np.argsort(self.event_times[ind])
        times = np.insert(self.event_times[ind][s], 0, 0)
        z = np.insert(self.total_rewards[ind][s].cumsum(), 0, 0)
        return times, z

    def mean_total_reward(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        s = np.argsort(self.event_times)
        times = np.insert(self.event_times[s], 0, 0)
        z = np.insert(self.total_rewards[s].cumsum(), 0, 0) / self.nb_samples
        return times, z
