from dataclasses import dataclass, field
from typing import NewType, Optional, Self, Sequence, Union

import numpy as np
from numpy.typing import NDArray

from relife.data import lifetime_data_factory


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
        args = None
        return IndexedLifetimeData(
            np.concatenate(
                [other.values[np.isin(other.index, inter_ids)] for other in others],
                axis=1,
            ),
            inter_ids,
            args,
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
        args = None
        sort_ind = np.argsort(
            index
        )  # FIXME: orders of the values seems to affects estimations of the parameters in Regression
        return IndexedLifetimeData(values[sort_ind], index[sort_ind], args)


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


FailureData = NewType("FailureData", Union[LifetimeData, NHPPData])
