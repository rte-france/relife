from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Measures:
    values: np.ndarray
    unit_ids: np.ndarray

    def __post_init__(self):
        if len(self.values.shape) != 2:
            raise ValueError("Invalid Measures values shape")
        if len(self.unit_ids.shape) != 1:
            raise ValueError("Invalid Measures unit_ids shape")
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
        Measures(values = np.array([[2, 3]]), unit_ids = np.array([10]))
    """

    def join_ids(*ids: np.ndarray):
        s = set.intersection(*map(lambda x: set(x), ids))
        return ids[0][masked_ids[0]], [np.in1d(_id, np.array(list(s))) for _id in ids]

    joined_ids, masked_ids = join_ids(*[m.unit_ids for m in measures])
    return Measures(np.hstack([m.values[masked_ids] for m in measures]), joined_ids)


def union_measures(*measures: Measures) -> Measures:
    """
    Args:
        *measures: Measures object.s containing values of shape (n1, p1), (n2, p2), etc.

    Returns:
        One Measures object where values are concatanation of every unit's values. The result
        is of shape (n1 + n2 + ..., p1 + p2 + ...).

    Examples:
        >>> measures_1 = Measures(values = np.array([[1], [2]]), unit_ids = np.array([3, 10]))
        >>> measures_2 = Measures(values = np.array([[3], [5]]), unit_ids = np.array([10, 2]))
        >>> join_measures(measures_1, measures_2)
        Measures(values = np.array([[2, 3]]), unit_ids = np.array([10]))
    """
    return Measures(
        np.hstack([m.values for m in measures]),
        np.hstack([m.unit_ids for m in measures]),
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
    def _sanity_checks(
            result: tuple[
                Measures, Measures, Measures, Measures, Measures, Measures, Measures
            ]
    ) -> None:
        (
            complete_lifetimes,
            left_censorships,
            right_censorships,
            interval_censorships,
            left_truncations,
            right_truncations,
            interval_truncations,
        ) = result

        if (
                interval_censorships.values[:, 0] >= interval_censorships.values[:, 1]
        ).any():
            raise ValueError("Invalid interval censorship values")

        if (
                interval_truncations.values[:, 0] >= interval_truncations.values[:, 1]
        ).any():
            raise ValueError("Invalid interval truncation values")

        joined_measures = join_measures(right_censorships, left_truncations)
        if (joined_measures.values[:, 0] <= joined_measures.values[:, 1]).any():
            raise ValueError(
                f"""
                right censored lifetime values can't be lower or equal to left
                truncation values:
                incompatible {right_censorships} and {left_truncations}
                """
            )

        joined_measures = join_measures(right_censorships, interval_truncations)
        if (joined_measures.values[:, 0] >= joined_measures.values[:, 2]).any():
            raise ValueError(
                f"""
                  right censored lifetime values can't be higer or equal to interval
                  of truncation: incompatible {right_censorships} and {interval_truncations}
                  """
            )

        joined_measures = join_measures(interval_censorships, interval_truncations)
        if (joined_measures.values[:, 0] < joined_measures.values[:, 2]).any():
            raise ValueError(
                f"""
                  interval censorship can't be outside of truncation interval:
                  incompatible {interval_censorships} and {interval_truncations}
                  """
            )
        elif (joined_measures.values[:, 1] > joined_measures.values[:, 3]).any():
            raise ValueError(
                f"""
                  interval censorship can't be outside of truncation interval:
                  incompatible {interval_censorships} and {interval_truncations}
                  """
            )

        joined_measures = join_measures(right_censorships, right_truncations)
        if (joined_measures.values[:, 0] >= joined_measures.values[:, 1]).any():
            raise ValueError(
                f"""
                  right censored lifetime values can't be higher than right truncations:
                  incompatible {right_censorships} and {right_truncations}
                  """
            )

        joined_measures = join_measures(left_censorships, right_truncations)
        if (joined_measures.values[:, 0] >= joined_measures.values[:, 1]).any():
            raise ValueError(
                f"""
                  left censored lifetime values can't be higher than right truncations:
                  incompatible {left_censorships} and {right_truncations}
                  """
            )

        joined_measures = join_measures(interval_censorships, right_truncations)
        if (joined_measures.values[:, 1] >= joined_measures.values[:, 2]).any():
            raise ValueError(
                f"""
                  interval censored lifetime values can't be higher than right truncations
                  : incompatible {interval_censorships} and {right_truncations}
                  """
            )

        joined_measures = join_measures(complete_lifetimes, right_truncations)
        if (joined_measures.values[:, 0] >= joined_measures.values[:, 1]).any():
            raise ValueError(
                f"""
                  complete lifetime values can't be higher than right truncations:
                  incompatible {complete_lifetimes} and {right_truncations}
                  """
            )

        joined_measures = join_measures(complete_lifetimes, left_truncations)
        if (joined_measures.values[:, 0] <= joined_measures.values[:, 1]).any():
            raise ValueError(
                f"""
                  complete lifetime values can't be lower than left truncations:
                  incompatible {complete_lifetimes} and {left_truncations}
                  """
            )

        joined_measures = join_measures(complete_lifetimes, interval_truncations)
        if (joined_measures.values[:, 0] >= joined_measures.values[:, 2]).any():
            raise ValueError(
                f"""
                  complete lifetime values can't be outside of truncation interval:
                  incompatible {complete_lifetimes} and {interval_truncations}
                  """
            )
        elif (joined_measures.values[:, 0] <= joined_measures.values[:, 1]).any():
            raise ValueError(
                f"""
                  complete lifetime values can't be outside of truncation interval:
                  incompatible {complete_lifetimes} and {interval_truncations}"""
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
            MeasuresParser._sanity_checks(result)
        except Exception as error:
            raise ValueError("Invalid input measures") from error

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

        if lc_indicators:
            lc_indicators = lc_indicators.astype(np.bool_, copy=False)
            if lc_indicators.shape != (n,):
                raise ValueError(f"invalid lc_indicators shape, expected ({n},)")
        else:
            lc_indicators = np.zeros_like(time, dtype=np.bool_)

        if rc_indicators:
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
        values = self.entry[index]
        return Measures(values, index)

    def get_right_truncations(self) -> Measures:
        index = np.where(self.departure != 0)[0]
        values = self.departure[index]
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
        values = self.entry[index]
        return Measures(values, index)

    def get_right_truncations(self) -> Measures:
        index = np.where(self.departure != 0)[0]
        values = self.departure[index]
        return Measures(values, index)

    def get_interval_truncations(self) -> Measures:
        index = np.where(np.logical_and(self.entry > 0, self.departure > 0))[0]
        values = np.hstack(
            (self.entry[index].reshape(-1, 1), self.departure[index].reshape(-1, 1))
        )
        return Measures(values, index)
