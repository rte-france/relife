from abc import ABC, abstractmethod
from typing import Generic, Optional, Sequence, TypeVarTuple, Union

import numpy as np
from numpy.typing import NDArray

from relife.data.base import IndexedLifetimeData, LifetimeData, NHPPData

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
