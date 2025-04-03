from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Protocol, Self

import numpy as np
from numpy.typing import NDArray


@dataclass
class IndexedLifetimeData:
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
            >>> data_1 = IndexedLifetimeData(values = np.array([[1.], [2.]]), index = np.array([3, 10]))
            >>> data_2 = IndexedLifetimeData(values = np.array([[3.], [5.]]), index = np.array([10, 2]))
            >>> data_1.intersection(data_2)
            IndexedData(values=array([[2, 3]]), index=array([10]))
        """

        inter_ids = np.array(
            list(set.intersection(*[set(other.index) for other in others]))
        ).astype(np.int64)
        return IndexedLifetimeData(
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
        return IndexedLifetimeData(values[sort_ind], index[sort_ind])


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
        self.rc = self.right_censoring.union(self.complete)
        self.rlc = self.complete.union(self.left_censoring, self.right_censoring)

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
        return IndexedLifetimeData(values, index)

    def get_left_censoring(self) -> IndexedLifetimeData:
        return IndexedLifetimeData(
            np.empty((0, 1), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
        )

    def get_right_censoring(self) -> IndexedLifetimeData:
        index = np.where(~self.event)[0]
        values = self.time[index]
        return IndexedLifetimeData(values, index)

    def get_interval_censoring(self) -> IndexedLifetimeData:
        rc_index = np.where(~self.event)[0]
        rc_values = np.c_[
            self.time[rc_index], np.ones(len(rc_index)) * np.inf
        ]  # add a column of inf
        return IndexedLifetimeData(rc_values, rc_index)

    def get_left_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return IndexedLifetimeData(values, index)

    def get_right_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return IndexedLifetimeData(values, index)


class Lifetime2DParser(LifetimeParser):
    """
    Concrete implementation of LifetimeDataReader for 2D encoding
    """

    def get_complete(self) -> IndexedLifetimeData:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index, 0]
        return IndexedLifetimeData(values, index)

    def get_left_censoring(
        self,
    ) -> IndexedLifetimeData:
        index = np.where(self.time[:, 0] == 0)[0]
        values = self.time[index, 1]
        return IndexedLifetimeData(values, index)

    def get_right_censoring(
        self,
    ) -> IndexedLifetimeData:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0]
        return IndexedLifetimeData(values, index)

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
        return IndexedLifetimeData(values, index)

    def get_left_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.entry > 0)[0]
        values = self.entry[index]
        return IndexedLifetimeData(values, index)

    def get_right_truncation(self) -> IndexedLifetimeData:
        index = np.where(self.departure < np.inf)[0]
        values = self.departure[index]
        return IndexedLifetimeData(values, index)


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
