from __future__ import annotations

import copy
from dataclasses import dataclass, field, InitVar
from itertools import product, zip_longest
from typing import TYPE_CHECKING, Optional, TypeVarTuple, Self

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from ._base import Likelihood
from relife.data import LifetimeData

if TYPE_CHECKING:
    from relife.lifetime_model import FittableParametricLifetimeModel

Args = TypeVarTuple("Args")


class LikelihoodFromLifetimes(Likelihood[*Args]):
    """
    Generic likelihood object for parametric_model lifetime model

    Parameters
    ----------
    model : ParametricLifetimeDistribution
        Underlying core used to compute probability functions
    lifetime_data : LifetimeData
        Observed lifetime data used one which the likelihood is evaluated
    """

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*Args],
        lifetime_data: LifetimeData,
    ):
        self.model = copy.deepcopy(model)
        self.data = StructuredLifetimeData(lifetime_data)

    @override
    @property
    def hasjac(self) -> bool:
        return hasattr(self.model, "jac_hf") and hasattr(self.model, "jac_chf")

    def _complete_contribs(self, structured_lifetime_data: StructuredLifetimeData) -> Optional[np.float64]:
        if structured_lifetime_data.complete is None:
            return None
        return -np.sum(
            np.log(
                self.model.hf(
                    structured_lifetime_data.complete.values,
                    *structured_lifetime_data.complete.args,
                )
            ) # (m, 1)
        ) # ()

    def _right_censored_contribs(self, structured_lifetime_data: StructuredLifetimeData) -> Optional[np.float64]:
        if structured_lifetime_data.complete_or_right_censored is None:
            return None
        return np.sum(
            self.model.chf(
                structured_lifetime_data.complete_or_right_censored.values,
                *structured_lifetime_data.complete_or_right_censored.args,
            ),
            dtype=np.float64, # (m, 1)
        ) # ()

    def _left_censored_contribs(self, structured_lifetime_data: StructuredLifetimeData) -> Optional[np.float64]:
        if structured_lifetime_data.left_censoring is None:
            return None
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.model.chf(
                        structured_lifetime_data.left_censoring.values,
                        *structured_lifetime_data.left_censoring.args,
                    )
                )
            ) # (m, 1)
        ) # ()

    def _left_truncations_contribs(
        self, structured_lifetime_data: StructuredLifetimeData
    ) -> Optional[np.float64]:
        if structured_lifetime_data.left_truncation is None:
            return None
        return -np.sum(
            self.model.chf(
                structured_lifetime_data.left_truncation.values,
                *structured_lifetime_data.left_truncation.args,
            ), # (m, 1)
            dtype=np.float64,
        ) # ()

    def _jac_complete_contribs(
        self, structured_lifetime_data: StructuredLifetimeData
    ) -> Optional[NDArray[np.float64]]:
        if structured_lifetime_data.complete is None:
            return None
        return -np.sum(
            self.model.jac_hf(
                structured_lifetime_data.complete.values,
                *structured_lifetime_data.complete.args,
                asarray=True,
            ) # (p, m, 1)
            / self.model.hf(
                structured_lifetime_data.complete.values,
                *structured_lifetime_data.complete.args,
            ), # (m, 1)
            axis=(1,2),
        ) # (p,)

    def _jac_right_censored_contribs(
        self, structured_lifetime_data: StructuredLifetimeData
    ) -> Optional[NDArray[np.float64]]:
        if structured_lifetime_data.complete_or_right_censored is None:
            return None
        return np.sum(
            self.model.jac_chf(
                structured_lifetime_data.complete_or_right_censored.values,
                *structured_lifetime_data.complete_or_right_censored.args,
                asarray=True,
            ), # (p, m, 1)
            axis=(1,2),
        ) # (p,)

    def _jac_left_censored_contribs(
        self, structured_lifetime_data: StructuredLifetimeData
    ) -> Optional[NDArray[np.float64]]:
        if structured_lifetime_data.left_censoring is None:
            return None
        return -np.sum(
            self.model.jac_chf(
                structured_lifetime_data.left_censoring.values,
                *structured_lifetime_data.left_censoring.args,
                asarray=True,
            ) # (p, m, 1)
            / np.expm1(
                self.model.chf(
                    structured_lifetime_data.left_censoring.values,
                    *structured_lifetime_data.left_censoring.args,
                )
            ), # (m, 1)
            axis=(1,2),
        ) # (p,)

    def _jac_left_truncations_contribs(
        self, structured_lifetime_data: StructuredLifetimeData
    ) -> Optional[NDArray[np.float64]]:
        if structured_lifetime_data.left_truncation is None:
            return None
        return -np.sum(
            self.model.jac_chf(
                structured_lifetime_data.left_truncation.values,
                *structured_lifetime_data.left_truncation.args,
                asarray=True,
            ), # (p, m, 1)
            axis=(1,2),
        ) # (p,)

    def negative_log(
        self,
        params: NDArray[np.float64], # (p,)
    ) -> np.float64:
        self.model.params = params
        contributions = (
            self._complete_contribs(self.data),
            self._right_censored_contribs(self.data),
            self._left_censored_contribs(self.data),
            self._left_truncations_contribs(self.data),
        )
        return sum(x for x in contributions if x is not None) # ()

    @override
    def jac_negative_log(
        self,
        params: NDArray[np.float64], # (p,)
    ) -> NDArray[np.float64]:
        if not self.hasjac:
            raise AttributeError(
                f"No support of jac negative likelihood for {self.model.__class__.__name__}"
            )
        self.model.params = params
        jac_contributions = (
            self._jac_complete_contribs(self.data),
            self._jac_right_censored_contribs(self.data),
            self._jac_left_censored_contribs(self.data),
            self._jac_left_truncations_contribs(self.data),
        )
        return sum(x for x in jac_contributions if x is not None) # (p,)


@dataclass
class IndexedLifetimeData:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: NDArray[np.float64] # (m, 1)
    index: NDArray[np.int64] # (m, 1)
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
class StructuredLifetimeData:
    """BLABLABLA"""

    lifetime_data : InitVar[LifetimeData]

    nb_samples: int = field(init=False)

    complete: Optional[IndexedLifetimeData] = field(repr=False, init=False)  # values shape (m, 1)
    right_censoring: Optional[IndexedLifetimeData] = field(
        repr=False, init=False
    )  # values shape (m, 1)
    left_censoring: Optional[IndexedLifetimeData] = field(
        repr=False, init=False
    )  # values shape (m, 1)
    interval_censoring: Optional[IndexedLifetimeData] = field(
        repr=False, init=False
    )  # values shape (m, 2)
    left_truncation: Optional[IndexedLifetimeData] = field(
        repr=False, init=False
    )  # values shape (m, 1)
    right_truncation: Optional[IndexedLifetimeData] = field(
        repr=False, init=False
    )  # values shape (m, 1)
    complete_or_right_censored: Optional[IndexedLifetimeData] = field(
        repr=False, init=False
    )

    def __len__(self):
        return self.nb_samples

    def __post_init__(self, lifetime_data : LifetimeData):

        self.nb_samples = len(lifetime_data)
        self.complete = StructuredLifetimeData.get_complete(lifetime_data)
        self.right_censoring = StructuredLifetimeData.get_right_censoring(lifetime_data),
        self.left_censoring = StructuredLifetimeData.get_left_censoring(lifetime_data),
        self.interval_censoring = StructuredLifetimeData.get_interval_censoring(lifetime_data),
        self.left_truncation = StructuredLifetimeData.get_left_truncation(lifetime_data),
        self.right_truncation = StructuredLifetimeData.get_right_truncation(lifetime_data),

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
            if data is not None and self.right_truncation is not None:
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
        self.complete_or_right_censored = None
        if self.complete is not None and self.right_censoring is not None:
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
        elif self.complete is not None:
            self.complete_or_right_censored = copy.deepcopy(self.complete)
        elif self.right_censoring is not None:
            self.complete_or_right_censored = copy.deepcopy(self.right_censoring)

    @staticmethod
    def get_complete(lifetime_data : LifetimeData) -> Optional[IndexedLifetimeData]:
        if lifetime_data.time.shape[-1] == 1: # 1D time
            index = np.where(lifetime_data.event)[0]
            if index.size > 0:
                values = lifetime_data.time[index]
                args = tuple((arg[index].copy() for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None
        else: # 2D time
            index = np.where(lifetime_data.time[:, 0] == lifetime_data.time[:, 1])[0]
            if index.size > 0:
                values = lifetime_data.time[index, 0]
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None

    @staticmethod
    def get_left_censoring(lifetime_data : LifetimeData) -> Optional[IndexedLifetimeData]:
        if lifetime_data.time.shape[-1] == 1: # 1D time
            return None
        else: # 2D time
            index = np.where(lifetime_data.time[:, 0] == 0)[0]
            if index.size > 0:
                values = lifetime_data.time[index, 1]
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None

    @staticmethod
    def get_right_censoring(lifetime_data : LifetimeData) -> Optional[IndexedLifetimeData]:
        if lifetime_data.time.shape[-1] == 1: # 1D time
            index = np.where(~lifetime_data.event)[0]
            if index.size > 0:
                values = lifetime_data.time[index]
                args = tuple((arg[index].copy() for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None
        else: # 2D time
            index = np.where(lifetime_data.time[:, 1] == np.inf)[0]
            if index.size > 0:
                values = lifetime_data.time[index, 0]
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None

    @staticmethod
    def get_interval_censoring(lifetime_data : LifetimeData) -> Optional[IndexedLifetimeData]:
        if lifetime_data.time.shape[-1] == 1: # 1D time
            rc_index = np.where(~lifetime_data.event)[0]
            if rc_index.size > 0:
                rc_values = np.c_[
                    lifetime_data.time[rc_index], np.ones(len(rc_index)) * np.inf
                ]  # add a column of inf
                args = tuple((arg[rc_index].copy() for arg in lifetime_data.args))
                return IndexedLifetimeData(rc_values, rc_index, args)
            return None
        else: # 2D time
            index = np.where(
                np.not_equal(lifetime_data.time[:, 0], lifetime_data.time[:, 1]),
            )[0]
            if index.size > 0:
                values = lifetime_data.time[index]
                if values.size != 0:
                    if np.any(values[:, 0] >= values[:, 1]):
                        raise ValueError(
                            "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                        )
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None

    @staticmethod
    def get_left_truncation(lifetime_data : LifetimeData) -> Optional[IndexedLifetimeData]:
        if lifetime_data.time.shape[-1] == 1: # 1D time
            index = np.where(lifetime_data.entry > 0)[0]
            if index.size > 0:
                values = lifetime_data.entry[index]
                args = tuple((arg[index].copy() for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None
        else: # 2D time
            index = np.where(lifetime_data.entry > 0)[0]
            if index.size > 0:
                values = lifetime_data.entry[index]
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None

    @staticmethod
    def get_right_truncation(lifetime_data : LifetimeData) -> Optional[IndexedLifetimeData]:
        if lifetime_data.time.shape[-1] == 1: # 1D time
            index = np.where(lifetime_data.departure < np.inf)[0]
            if index.size > 0:
                values = lifetime_data.departure[index]
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None
        else: # 2D time
            index = np.where(lifetime_data.departure < np.inf)[0]
            if index.size > 0:
                values = lifetime_data.departure[index]
                args = tuple((arg[index] for arg in lifetime_data.args))
                return IndexedLifetimeData(values, index, args)
            return None