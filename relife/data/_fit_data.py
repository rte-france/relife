from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import product, zip_longest
from typing import Generic, NewType, Optional, Self, Sequence, TypeVarTuple, Union

import numpy as np
from numpy.typing import NDArray


@dataclass
class IndexedLifetimeData:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: NDArray[np.float64]
    index: NDArray[np.int64]
    args: Optional[tuple[float | NDArray[np.float64], ...]] = field(
        repr=False, default_factory=tuple
    )

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

        other_args = tuple(
            (np.concatenate(x) for x in product(*(other.args for other in others)))
        )
        args = tuple((np.concatenate(x) for x in product(self.args, other_args)))

        sort_ind = np.argsort(
            index
        )  # FIXME: orders of the values seems to affects estimations of the parameters in Regression

        return IndexedLifetimeData(
            values[sort_ind], index[sort_ind], tuple((arg[index] for arg in args))
        )


@dataclass
class LifetimeData:
    """BLABLABLA"""

    nb_samples: int
    complete: IndexedLifetimeData = field(repr=False)  # values shape (m, 1)
    left_censoring: IndexedLifetimeData = field(repr=False)  # values shape (m, 1)
    right_censoring: IndexedLifetimeData = field(repr=False)  # values shape (m, 1)
    interval_censoring: IndexedLifetimeData = field(repr=False)  # values shape (m, 2)
    left_truncation: IndexedLifetimeData = field(repr=False)  # values shape (m, 1)
    right_truncation: IndexedLifetimeData = field(repr=False)  # values shape (m, 1)

    def __len__(self):
        return self.nb_samples

    def __post_init__(self):
        # sanity checks that observed lifetimes are inside truncation bounds
        for field_name in [
            "complete",
            "left_censoring",
            "left_censoring",
            "interval_censoring",
        ]:
            data = getattr(self, field_name)
            if len(self.left_truncation) != 0 and len(data) != 0:
                inter_ids = (np.intersect1d(data.index, self.left_truncation.index),)
                intersection_values = np.concatenate(
                    (
                        data.values[np.isin(data.index, inter_ids)],
                        self.left_truncation.values[
                            np.isin(self.left_truncation.index, inter_ids)
                        ],
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
                        < intersection_values[
                            :, [-1]
                        ]  # then check if any is under left truncation bound
                    ):
                        raise ValueError(
                            "Some lifetimes are under left truncation bounds"
                        )
            if len(self.right_truncation) != 0 and len(data) != 0:

                inter_ids = np.intersect1d(data.index, self.right_truncation.index)
                intersection_values = np.concatenate(
                    (
                        data.values[np.isin(data.index, inter_ids)],
                        self.right_truncation.values[
                            np.isin(self.right_truncation.index, inter_ids)
                        ],
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
                        > intersection_values[
                            :, [-1]
                        ]  # then check if any is above right truncation bound
                    ):
                        raise ValueError(
                            "Some lifetimes are above right truncation bounds"
                        )

        # compute complete_or_right_censored
        values = np.concatenate(
            [self.complete.values, self.right_censoring.values], axis=0
        )
        index = np.concatenate(
            [self.complete.index, self.right_censoring.index], axis=0
        )
        args = tuple(
            (
                np.concatenate(x, axis=0)
                for x in zip_longest(self.complete.args, self.right_censoring.args)
            )
        )
        # FIXME: orders of the values seems to affects estimations of the parameters in Regression
        sort_index = np.argsort(index)

        self.complete_or_right_censored = IndexedLifetimeData(
            values[sort_index],
            index[sort_index],
            tuple((arg[sort_index] for arg in args)),
        )


@dataclass
class NHPPData:
    events_assets_ids: Union[Sequence[str], NDArray[np.int64]]
    ages: NDArray[np.float64]
    assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = field(
        repr=False, default=None
    )
    first_ages: Optional[NDArray[np.float64]] = field(repr=False, default=None)
    last_ages: Optional[NDArray[np.float64]] = field(repr=False, default=None)
    args: Optional[tuple[float | NDArray[np.float64], ...]] = field(
        repr=False, default=None
    )

    first_age_index: NDArray[np.int64] = field(repr=False, init=False)
    last_age_index: NDArray[np.int64] = field(repr=False, init=False)

    def __post_init__(self):
        # sort fields
        sort_ind = np.lexsort((self.ages, self.events_assets_ids))
        events_assets_ids = self.events_assets_ids[sort_ind]
        ages = self.ages[sort_ind]

        # number of age value per asset id
        nb_ages_per_asset = np.unique_counts(events_assets_ids).counts
        # index of the first ages and last ages in ages_at_event
        self.first_age_index = np.where(
            np.roll(events_assets_ids, 1) != events_assets_ids
        )[0]
        self.last_age_index = np.append(
            self.first_age_index[1:] - 1, len(events_assets_ids) - 1
        )

        if self.assets_ids is not None:

            # sort fields
            sort_ind = np.sort(self.assets_ids)
            self.first_ages = (
                self.first_ages[sort_ind]
                if self.first_ages is not None
                else self.first_ages
            )
            self.last_ages = (
                self.last_ages[sort_ind]
                if self.last_ages is not None
                else self.last_ages
            )
            self.args = tuple((arg[sort_ind] for arg in self.args))

            if self.first_ages is not None:
                if np.any(
                    ages[self.first_age_index]
                    <= self.first_ages[nb_ages_per_asset != 0]
                ):
                    raise ValueError(
                        "Each start_ages value must be lower than all of its corresponding ages_at_event values"
                    )
            if self.last_ages is not None:
                if np.any(
                    ages[self.last_age_index] >= self.last_ages[nb_ages_per_asset != 0]
                ):
                    raise ValueError(
                        "Each end_ages value must be greater than all of its corresponding ages_at_event values"
                    )

    def to_lifetime_data(self) -> LifetimeData:
        event = np.ones_like(self.ages, dtype=np.bool_)
        # insert_index = np.cumsum(nb_ages_per_asset)
        # insert_index = last_age_index + 1
        if self.last_ages is not None:
            time = np.insert(self.ages, self.last_age_index + 1, self.last_ages)
            event = np.insert(event, self.last_age_index + 1, False)
            _ids = np.insert(
                self.events_assets_ids, self.last_age_index + 1, self.assets_ids
            )
            if self.first_ages is not None:
                entry = np.insert(
                    self.ages,
                    np.insert((self.last_age_index + 1)[:-1], 0, 0),
                    self.first_ages,
                )
            else:
                entry = np.insert(self.ages, self.first_age_index, 0.0)
        else:
            time = self.ages.copy()
            _ids = self.events_assets_ids.copy()
            if self.first_ages is not None:
                entry = np.roll(self.ages, 1)
                entry[self.first_age_index] = self.first_ages
            else:
                entry = np.roll(self.ages, 1)
                entry[self.first_age_index] = 0.0
        model_args = tuple((np.take(arg, _ids) for arg in self.args))

        return lifetime_data_factory(time, *model_args, event=event, entry=entry)


Args = TypeVarTuple("Args")


class LifetimeParser(Generic[*Args], ABC):
    """
    Factory method of ObservedLifetimes and Truncations
    """

    def __init__(
        self,
        time: NDArray[np.float64],
        /,
        *args: *Args,
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
        self.args = args

    @abstractmethod
    def get_complete(self) -> IndexedLifetimeData:
        """
        Returns:
            IndexedLifetimeData: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censoring(self) -> IndexedLifetimeData:
        """
        Returns:
            IndexedLifetimeData: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censoring(self) -> IndexedLifetimeData:
        """
        Returns:
            IndexedLifetimeData: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censoring(self) -> IndexedLifetimeData:
        """
        Returns:
            IndexedLifetimeData: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncation(self) -> IndexedLifetimeData:
        """
        Returns:
            IndexedLifetimeData: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncation(self) -> IndexedLifetimeData:
        """
        Returns:
            IndexedLifetimeData: object containing right truncations values and index
        """


class Lifetime1DParser(LifetimeParser):
    """
    Concrete implementation of LifetimeDataReader for 1D encoding
    """

    def get_complete(self) -> IndexedLifetimeData:
        index = np.where(self.event)[0]
        values = self.time[index]
        args = tuple((arg[index].copy() for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_left_censoring(self) -> IndexedLifetimeData:
        return IndexedLifetimeData(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    def get_right_censoring(self) -> IndexedLifetimeData:
        index = np.where(~self.event)[0]
        values = self.time[index]
        args = tuple((arg[index].copy() for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_interval_censoring(self) -> IndexedLifetimeData:
        rc_index = np.where(~self.event)[0]
        rc_values = np.c_[
            self.time[rc_index], np.ones(len(rc_index)) * np.inf
        ]  # add a column of inf
        args = tuple((arg[rc_index].copy() for arg in self.args))
        return IndexedLifetimeData(rc_values, rc_index, args)

    def get_left_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        args = tuple((arg[index].copy() for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_right_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)


class Lifetime2DParser(LifetimeParser):
    """
    Concrete implementation of LifetimeDataReader for 2D encoding
    """

    def get_complete(self) -> IndexedLifetimeData:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0]
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_left_censoring(
        self,
    ) -> IndexedLifetimeData:
        index = np.where(self.time[:, 0] == 0)[0]
        values = self.time[index, 1]
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_right_censoring(
        self,
    ) -> IndexedLifetimeData:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0]
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_interval_censoring(self) -> IndexedLifetimeData:
        index = np.where(
            np.not_equal(self.time[:, 0], self.time[:, 1]),
        )[0]
        values = self.time[index]
        if values.size != 0:
            if np.any(values[:, 0] >= values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_left_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)

    def get_right_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        args = tuple((arg[index] for arg in self.args))
        return IndexedLifetimeData(values, index, args)


Args = TypeVarTuple("Args")


def lifetime_data_factory(
    time: NDArray[np.float64],
    /,
    *args: *Args,
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
        reader = Lifetime1DParser(
            time, *args, event=event, entry=entry, departure=departure
        )
    elif time.ndim == 2:
        if time.shape[-1] != 2:
            raise ValueError("If time ndim is 2, time shape must be (n, 2)")
        reader = Lifetime2DParser(
            time, *args, event=event, entry=entry, departure=departure
        )
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


def nhpp_data_factory(
    events_assets_ids: Union[Sequence[str], NDArray[np.int64]],
    ages: NDArray[np.float64],
    /,
    *args: *Args,
    assets_ids: Optional[Union[Sequence[str], NDArray[np.int64]]] = None,
    first_ages: Optional[NDArray[np.float64]] = None,
    last_ages: Optional[NDArray[np.float64]] = None,
) -> NHPPData:
    # convert inputs to arrays
    events_assets_ids = np.asarray(events_assets_ids)
    ages = np.asarray(ages, dtype=np.float64)
    if assets_ids is not None:
        assets_ids = np.asarray(assets_ids)
    if first_ages is not None:
        first_ages = np.asarray(first_ages, dtype=np.float64)
    if last_ages is not None:
        last_ages = np.asarray(last_ages, dtype=np.float64)

    # control shapes
    if events_assets_ids.ndim != 1:
        raise ValueError("Invalid array shape for events_assets_ids. Expected 1d-array")
    if ages.ndim != 1:
        raise ValueError("Invalid array shape for ages_at_event. Expected 1d-array")
    if len(events_assets_ids) != len(ages):
        raise ValueError(
            "Shape of events_assets_ids and ages_at_event must be equal. Expected equal length 1d-arrays"
        )
    if assets_ids is not None:
        if assets_ids.ndim != 1:
            raise ValueError("Invalid array shape for assets_ids. Expected 1d-array")
        if first_ages is not None:
            if first_ages.ndim != 1:
                raise ValueError(
                    "Invalid array shape for start_ages. Expected 1d-array"
                )
            if len(first_ages) != len(assets_ids):
                raise ValueError(
                    "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"
                )
        if last_ages is not None:
            if last_ages.ndim != 1:
                raise ValueError("Invalid array shape for end_ages. Expected 1d-array")
            if len(last_ages) != len(assets_ids):
                raise ValueError(
                    "Shape of assets_ids and end_ages must be equal. Expected equal length 1d-arrays"
                )
        if bool(args):
            for arg in args:
                arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                if arg.ndim > 2:
                    raise ValueError(
                        "Invalid arg shape in model_args. Arrays must be 0, 1 or 2d"
                    )
                try:
                    arg.reshape((len(assets_ids), -1))
                except ValueError:
                    raise ValueError(
                        "Invalid arg shape in model_args. Arrays must coherent with the number of assets given by assets_ids"
                    )
    else:
        if first_ages is not None:
            raise ValueError(
                "If start_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if last_ages is not None:
            raise ValueError(
                "If end_ages is given, corresponding asset ids must be given in assets_ids"
            )
        if bool(args):
            raise ValueError(
                "If model_args is given, corresponding asset ids must be given in assets_ids"
            )

    if events_assets_ids.dtype != np.int64:
        events_assets_ids = np.unique(events_assets_ids, return_inverse=True)[1]
    # convert assets_id to int id
    if assets_ids is not None:
        if assets_ids.dtype != np.int64:
            assets_ids = np.unique(assets_ids, return_inverse=True)[1]
        # control ids correspondance
        if not np.all(np.isin(events_assets_ids, assets_ids)):
            raise ValueError(
                "If assets_ids is filled, all values of events_assets_ids must exist in assets_ids"
            )

    return NHPPData(events_assets_ids, ages, first_ages, last_ages, args)


FailureData = NewType("FailureData", Union[LifetimeData, NHPPData])
