"""
This module gathers class used by subsystems' objects to treat lifetime data
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Measures:
    """
    Object that encapsulates lifetime data values and corresponding units index
    """

    values: np.ndarray
    unit_ids: np.ndarray

    def __post_init__(self):
        if self.values.ndim != 2:
            raise ValueError("Invalid Measures values number of dimensions")
        if self.unit_ids.ndim != 1:
            raise ValueError("Invalid Measures unit_ids number of dimensions")
        if len(self.values) != len(self.unit_ids):
            raise ValueError("Incompatible Measures values and unit_ids")

    def __len__(self) -> int:
        return len(self.values)


def intersect_measures(*measures: Measures) -> Measures:
    """
    Args:
        *measures: Measures object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:
        Measures: One Measures object where values are concatanation of common units values. The result
        is of shape (N, p1 + p2 + ...).

    Examples:
        >>> measures_1 = Measures(values = np.array([[1], [2]]), unit_ids = np.array([3, 10]))
        >>> measures_2 = Measures(values = np.array([[3], [5]]), unit_ids = np.array([10, 2]))
        >>> intersect_measures(measures_1, measures_2)
        Measures(values=array([[2, 3]]), unit_ids=array([10]))
    """

    inter_ids = np.array(list(set.intersection(*[set(m.unit_ids) for m in measures])))
    return Measures(
        np.hstack([m.values[np.isin(m.unit_ids, inter_ids)] for m in measures]),
        inter_ids,
    )


class MeasuresParser(ABC):
    """
    Factory method of Measures object
    """

    def __init__(
        self,
        time: np.ndarray,
        lc_indicators: np.ndarray = None,
        rc_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
    ):
        if lc_indicators is None:
            lc_indicators = np.zeros_like(time, dtype=np.bool_)
        if rc_indicators is None:
            rc_indicators = np.zeros_like(time, dtype=np.bool_)
        if entry is None:
            entry = np.zeros_like(time)
        if departure is None:
            departure = np.ones_like(time) * np.inf

        self.time = time
        self.lc_indicators = lc_indicators.astype(np.bool_, copy=False)
        self.rc_indicators = rc_indicators.astype(np.bool_, copy=False)
        self.entry = entry
        self.departure = departure
        self._check_format()

        if np.any(np.logical_and(self.lc_indicators, self.rc_indicators)) is True:
            raise ValueError(
                """
                lc_indicators and rc_indicators can't be true at the same index
                """
            )

    def _check_format(self) -> None:
        (n, _) = self.time.shape
        for values in (
            self.entry,
            self.departure,
            self.lc_indicators,
            self.rc_indicators,
        ):
            if values.shape != (n,):
                raise ValueError(f"invalid argument shape, expected ({n},)")

    @abstractmethod
    def get_complete(self) -> Measures:
        """
        Returns:
            Measures: object containing complete lifetime values and index
        """

    @abstractmethod
    def get_left_censorships(self) -> Measures:
        """
        Returns:
            Measures: object containing left censorhips values and index
        """

    @abstractmethod
    def get_right_censorships(self) -> Measures:
        """
        Returns:
            Measures: object containing right censorhips values and index
        """

    @abstractmethod
    def get_interval_censorships(self) -> Measures:
        """
        Returns:
            Measures: object containing interval censorhips valuess and index
        """

    @abstractmethod
    def get_left_truncations(self) -> Measures:
        """
        Returns:
            Measures: object containing left truncations values and index
        """

    @abstractmethod
    def get_right_truncations(self) -> Measures:
        """
        Returns:
            Measures: object containing right truncations values and index
        """

    @staticmethod
    def _compatible_with_left_truncations(
        lifetimes: Measures, left_truncations: Measures
    ) -> None:
        if len(lifetimes) != 0 and len(left_truncations) != 0:
            intersected_measures = intersect_measures(lifetimes, left_truncations)
            if len(intersected_measures) != 0:
                if np.any(
                    np.min(
                        intersected_measures.values[:, : lifetimes.values.shape[-1]],
                        axis=1,
                        keepdims=True,
                    )
                    < intersected_measures.values[:, lifetimes.values.shape[-1] :]
                ):
                    raise ValueError(
                        f"""
                        Some lifetimes are under left truncation bounds :
                        {lifetimes} and {left_truncations}
                        """
                    )

    @staticmethod
    def _compatible_with_right_truncations(
        lifetimes: Measures, right_truncations: Measures
    ) -> None:
        if len(lifetimes) != 0 and len(right_truncations) != 0:
            intersected_measures = intersect_measures(lifetimes, right_truncations)
            if len(intersected_measures) != 0:
                if np.any(
                    np.max(
                        intersected_measures.values[:, : lifetimes.values.shape[-1]],
                        axis=1,
                        keepdims=True,
                    )
                    > intersected_measures.values[:, lifetimes.values.shape[-1] :]
                ):
                    raise ValueError(
                        f"""
                        Some lifetimes are above right truncation bounds :
                        {lifetimes} and {right_truncations}
                        """
                    )

    def __call__(
        self,
    ) -> tuple[Measures, Measures, Measures, Measures, Measures, Measures]:
        result = (
            self.get_complete(),
            self.get_left_censorships(),
            self.get_right_censorships(),
            self.get_interval_censorships(),
            self.get_left_truncations(),
            self.get_right_truncations(),
        )
        try:
            for lifetimes in result[:4]:
                MeasuresParser._compatible_with_left_truncations(lifetimes, result[4])
                MeasuresParser._compatible_with_right_truncations(lifetimes, result[5])
        except Exception as error:
            raise ValueError("Incorrect input measures") from error
        return result


class MeasuresParserFrom1D(MeasuresParser):
    """
    Concrete implementation of MeasuresParser factory for 1D encoding
    """

    def get_complete(self) -> Measures:
        index = np.where(np.logical_and(~self.lc_indicators, ~self.rc_indicators))[0]
        values = self.time[index].reshape(-1, 1)
        return Measures(values, index)

    def get_left_censorships(self) -> Measures:
        index = np.where(self.lc_indicators)[0]
        values = self.time[index].reshape(-1, 1)
        return Measures(values, index)

    def get_right_censorships(self) -> Measures:
        index = np.where(self.rc_indicators)[0]
        values = self.time[index].reshape(-1, 1)
        return Measures(values, index)

    def get_interval_censorships(self) -> Measures:
        return Measures(np.empty((0, 2)), np.empty((0,)))

    def get_left_truncations(self) -> Measures:
        index = np.where(self.entry != 0)[0]
        values = self.entry[index].reshape(-1, 1)
        return Measures(values, index)

    def get_right_truncations(self) -> Measures:
        index = np.where(self.departure != 0)[0]
        values = self.departure[index].reshape(-1, 1)
        return Measures(values, index)


class MeasuresParserFrom2D(MeasuresParser):
    """
    Concrete implementation of MeasuresParser factory for 2D encoding
    """

    def get_complete(self) -> Measures:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index][:, 0].reshape(-1, 1)
        return Measures(values, index)

    def get_left_censorships(
        self,
    ) -> Measures:
        index = np.where(self.time[:, 0] == 0.0)[0]
        values = self.time[index, 1].reshape(-1, 1)
        return Measures(values, index)

    def get_right_censorships(
        self,
    ) -> Measures:
        index = np.where(self.time[:, 1] == np.inf)[0]
        values = self.time[index, 0].reshape(-1, 1)
        return Measures(values, index)

    def get_interval_censorships(self) -> Measures:
        index = np.where(
            np.logical_and(
                np.logical_and(
                    self.time[:, 0] > 0,
                    self.time[:, 1] < np.inf,
                ),
                np.not_equal(self.time[:, 0], self.time[:, 1]),
            )
        )[0]
        values = self.time[index]
        measures = Measures(values, index)
        if len(measures) != 0:
            if np.any(measures.values[:, 0] >= measures.values[:, 1]):
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )
        return measures

    def get_left_truncations(self) -> Measures:
        index = np.where(self.entry != 0)[0]
        values = self.entry[index].reshape(-1, 1)
        return Measures(values, index)

    def get_right_truncations(self) -> Measures:
        index = np.where(self.departure != 0)[0]
        values = self.departure[index].reshape(-1, 1)
        return Measures(values, index)
