from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

#### si je lance ça va fonctionner (sans mes modifs)
# mettre dans likelif=hood le choix entre les parser... (dans backbone, Likelikelihood)


@dataclass(
    frozen=True
)  # décorateur => peux vérifier que deux instance sont égales (pas faisable avec autres classes)
class Measures:  # objet qui encapsule 2 attibuts : values et unit_ids
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
        One Measures object where values are concatanation of common units values. The result
        is of shape (N, p1 + p2 + ...).

    Examples:
        >>> measures_1 = Measures(values = np.array([[1], [2]]), unit_ids = np.array([3, 10]))
        >>> measures_2 = Measures(values = np.array([[3], [5]]), unit_ids = np.array([10, 2]))
        >>> intersect_measures(measures_1, measures_2)
        Measures(values=array([[2, 3]]), unit_ids=array([10]))
    """

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
    ) -> tuple[
        Measures, Measures, Measures, Measures, Measures, Measures
    ]:  # result return has 6 Measures
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

        if np.any(np.logical_and(lc_indicators, rc_indicators)) is True:
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

    # def __call__(self):
    #     MeasuresParser.__call__(self)

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
    def __init__(
        self, time: np.ndarray, entry: np.ndarray = None, departure: np.ndarray = None
    ):
        (n, p) = time.shape

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
        self.entry = entry
        self.departure = departure

    def get_complete(self) -> Measures:
        index = np.where(self.time[:, 0] == self.time[:, 1])[0]
        values = self.time[index][:, 0].reshape(-1, 1)
        return Measures(values, index)

    # def __call__(self):
    #     MeasuresParser.__call__(self)

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
