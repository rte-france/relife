from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Measures:
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


def join_measures(*measures: Measures) -> Measures:
    """
    Args:
        *measures: Measures object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:
        One Measures object where values are concatanation of common units values. The result
        is of shape (N, p1 + p2 + ...).

    Examples:
        >>> measures_1 = Measures(values = np.array([[1], [2]]), unit_ids = np.array([3, 10]))
        >>> measures_2 = Measures(values = np.array([[3], [5]]), unit_ids = np.array([10, 2]))
        >>> join_measures(measures_1, measures_2)
        Measures(values=array([[2, 3]]), unit_ids=array([10]))
    """

    # def join_ids(*ids: np.ndarray):
    #     s = set.intersection(*map(lambda x: set(x), ids))
    #     masked_ids = [np.in1d(_id, np.array(list(s))) for _id in ids]
    #     return ids[0][masked_ids[0]], masked_ids

    inter_ids = np.array(
        list(set.intersection(*map(lambda x: set(x), [m.unit_ids for m in measures])))
    )
    return Measures(
        np.hstack([m.values[np.isin(m.unit_ids, inter_ids)] for m in measures]),
        inter_ids,
    )


class MeasuresParser(ABC):
    """
    Factory method
    """

    @abstractmethod
    def get_complete(self) -> Measures:
        pass

    @abstractmethod
    def get_left_censorships(self) -> Measures:
        pass

    @abstractmethod
    def get_right_censorships(self) -> Measures:
        pass

    @abstractmethod
    def get_interval_censorships(self) -> Measures:
        pass

    @abstractmethod
    def get_left_truncations(self) -> Measures:
        pass

    @abstractmethod
    def get_right_truncations(self) -> Measures:
        pass

    @abstractmethod
    def get_interval_truncations(self) -> Measures:
        pass

    @staticmethod
    def _check_interval_censorships(interval_censorships: Measures) -> None:
        if len(interval_censorships) != 0:
            if (
                    interval_censorships.values[:, 0] >= interval_censorships.values[:, 1]
            ).any():
                raise ValueError(
                    "Interval censorships lower bounds can't be higher or equal to its upper bounds"
                )

    @staticmethod
    def _check_interval_truncations(interval_truncations: Measures) -> None:
        if len(interval_truncations) != 0:
            if (
                    interval_truncations.values[:, 0] >= interval_truncations.values[:, 1]
            ).any():
                raise ValueError(
                    "Interval truncations lower bounds can't be higher or equal to its upper bounds"
                )

    @staticmethod
    def _compatible_with_left_truncations(
            lifetimes: Measures, left_truncations: Measures
    ) -> None:
        if len(lifetimes) != 0 and len(left_truncations) != 0:
            joined_measures = join_measures(lifetimes, left_truncations)
            if len(joined_measures) != 0:
                if (
                        np.min(
                            joined_measures.values[:, : lifetimes.values.shape[-1]],
                            axis=1,
                            keepdims=True,
                        )
                        < joined_measures.values[:, lifetimes.values.shape[-1]:]
                ).any():
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
            joined_measures = join_measures(lifetimes, right_truncations)
            if len(joined_measures) != 0:
                if (
                        np.max(
                            joined_measures.values[:, : lifetimes.values.shape[-1]],
                            axis=1,
                            keepdims=True,
                        )
                        > joined_measures.values[:, 1]
                ).any():
                    raise ValueError(
                        f"""
                        Some lifetimes are above right truncation bounds :
                        {lifetimes} and {right_truncations}
                        """
                    )

    @staticmethod
    def _compatible_with_interval_truncations(
            lifetimes: Measures, interval_truncations: Measures
    ) -> None:
        if len(lifetimes) != 0 and len(interval_truncations) != 0:
            joined_measures = join_measures(lifetimes, interval_truncations)
            if len(joined_measures) != 0:
                if (
                        np.max(
                            joined_measures.values[:, : lifetimes.values.shape[-1]],
                            axis=1,
                            keepdims=True,
                        )
                        > np.max(
                    joined_measures.values[:, lifetimes.values.shape[-1]:],
                    axis=1,
                    keepdims=True,
                )
                ).any():
                    raise ValueError(
                        f"""
                        Some lifetimes above interval truncation bounds :
                        {lifetimes} and {interval_truncations}
                        """
                    )
                elif (
                        np.min(
                            joined_measures.values[:, : lifetimes.values.shape[-1]],
                            axis=1,
                            keepdims=True,
                        )
                        < np.min(
                    joined_measures.values[:, lifetimes.values.shape[-1]:],
                    axis=1,
                    keepdims=True,
                )
                ).any():
                    raise ValueError(
                        f"""
                        Some lifetimes below interval truncation bounds :
                        {lifetimes} and {interval_truncations}
                        """
                    )

    def __call__(
            self,
    ) -> tuple[Measures, Measures, Measures, Measures, Measures, Measures, Measures]:
        result = (
            self.get_complete(),
            self.get_left_censorships(),
            self.get_right_censorships(),
            self.get_interval_censorships(),
            self.get_left_truncations(),
            self.get_right_truncations(),
            self.get_interval_truncations(),
        )
        try:
            MeasuresParser._check_interval_censorships(result[3])
            MeasuresParser._check_interval_truncations(result[6])
            for lifetimes in result[:4]:
                MeasuresParser._compatible_with_left_truncations(lifetimes, result[4])
                MeasuresParser._compatible_with_right_truncations(lifetimes, result[5])
                MeasuresParser._compatible_with_right_truncations(lifetimes, result[6])
        except Exception as error:
            raise ValueError("Incorrect input measures") from error

        return result


class MeasuresParserFrom1D(MeasuresParser):
    def __init__(
            self,
            time: np.ndarray,
            lc_indicators: np.ndarray = None,
            rc_indicators: np.ndarray = None,
            entry: np.ndarray = None,
            departure: np.ndarray = None,
    ):

        (n,) = time.shape

        if lc_indicators is not None:
            lc_indicators = lc_indicators.astype(np.bool_, copy=False)
            if lc_indicators.shape != (n,):
                raise ValueError(f"invalid lc_indicators shape, expected ({n},)")
        else:
            lc_indicators = np.zeros_like(time, dtype=np.bool_)

        if rc_indicators is not None:
            rc_indicators = rc_indicators.astype(np.bool_, copy=False)
            if rc_indicators.shape != (n,):
                raise ValueError(f"invalid rc_indicators shape, expected ({n},)")
        else:
            rc_indicators = np.zeros_like(time, dtype=np.bool_)

        if np.logical_and(lc_indicators, rc_indicators).any() is True:
            raise ValueError(
                """
                lc_indicators and rc_indicators can't be true at the same index
                """
            )

        if entry is not None:
            if entry.shape != (n,):
                raise ValueError(f"invalid entry shape, expected ({n},)")
        else:
            entry = np.empty((0, 1), dtype=float)

        if departure is not None:
            if departure.shape != (n,):
                raise ValueError(f"invalid departure shape, expected ({n},)")
        else:
            departure = np.empty((0, 1), dtype=float)

        self.time = time
        self.lc_indicators = lc_indicators
        self.rc_indicators = rc_indicators
        self.entry = entry
        self.departure = departure

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

    def get_interval_truncations(self) -> Measures:
        index = np.where(np.logical_and(self.entry > 0, self.departure > 0))[0]
        values = np.hstack(
            (self.entry[index].reshape(-1, 1), self.departure[index].reshape(-1, 1))
        )
        return Measures(values, index)


class MeasuresParserFrom2D(MeasuresParser):
    def __init__(
            self, time: np.ndarray, entry: np.ndarray = None, departure: np.ndarray = None
    ):
        (n, p) = time.shape

        if entry:
            if entry.shape != (n,):
                raise ValueError(f"invalid entry shape, expected ({n},)")
        else:
            entry = np.empty((0, 1), dtype=float)

        if departure:
            if departure.shape != (n,):
                raise ValueError(f"invalid departure shape, expected ({n},)")
        else:
            departure = np.empty((0, 1), dtype=float)

        self.time = time
        self.entry = entry
        self.departure = departure

    def get_complete(self) -> Measures:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index][:, 0].reshape(-1, 1)
        return Measures(values, index)

    def get_left_censorships(self) -> Measures:
        index = np.where(self.time[:, 0] == 0.0)[0]
        values = self.time[index, 1].reshape(-1, 1)
        return Measures(values, index)

    def get_right_censorships(self) -> Measures:
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
        return Measures(values, index)

    def get_left_truncations(self) -> Measures:
        index = np.where(self.entry != 0)[0]
        values = self.entry[index].reshape(-1, 1)
        return Measures(values, index)

    def get_right_truncations(self) -> Measures:
        index = np.where(self.departure != 0)[0]
        values = self.departure[index].reshape(-1, 1)
        return Measures(values, index)

    def get_interval_truncations(self) -> Measures:
        index = np.where(np.logical_and(self.entry > 0, self.departure > 0))[0]
        values = np.hstack(
            (self.entry[index].reshape(-1, 1), self.departure[index].reshape(-1, 1))
        )
        return Measures(values, index)
