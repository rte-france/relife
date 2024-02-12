from abc import ABC, abstractmethod

import numpy as np


class LifetimeFormat(ABC):
    def __init__(self, values: np.ndarray):

        self.regular_index = np.where(np.ones(len(values), dtype=bool))[0]
        self.left_index = np.where(np.zeros(len(values), dtype=bool))[0]
        self.right_index = np.where(np.zeros(len(values), dtype=bool))[0]
        self.interval_index = np.where(np.zeros(len(values), dtype=bool))[0]

    @abstractmethod
    def set_left_index(self, **kwargs):
        pass

    @abstractmethod
    def set_right_index(self, **kwargs):
        pass

    @abstractmethod
    def set_interval_index(self, **kwargs):
        pass

    @abstractmethod
    def set_regular_index(self, **kwargs):
        pass

    @abstractmethod
    def set_left_values(self, **kwargs):
        pass

    @abstractmethod
    def set_right_values(self, **kwargs):
        pass

    @abstractmethod
    def set_interval_values(self, **kwargs):
        pass

    @abstractmethod
    def set_regular_values(self, **kwargs):
        pass

    def build(self, **kwargs):
        self.set_left_index(**kwargs)
        self.set_right_index(**kwargs)
        self.set_interval_index(**kwargs)
        self.set_regular_index(**kwargs)
        self.set_left_values(**kwargs)
        self.set_right_values(**kwargs)
        self.set_interval_values(**kwargs)
        self.set_regular_values(**kwargs)


class BaseCensoredLifetime(LifetimeFormat):
    def __init__(self, values=np.ndarray):
        super().__init__(values)

    @staticmethod
    def _check_indicators(values: np.ndarray, indicators: np.ndarray):
        if type(indicators) == np.ndarray:
            if indicators.size != 0:
                assert len(indicators.shape) == 1, "indicators must be 1d array"
                assert len(indicators) == len(
                    values
                ), "indicators must have the same length as lifetime values"
                if indicators.dtype != np.bool_:
                    indicators = indicators.astype(bool)
        else:
            ValueError("indicators must be np.ndarray")
        return indicators

    def set_left_index(
        self,
        values: np.ndarray,
        left_indicators: np.ndarray = np.array([], dtype=int),
        **kwargs,
    ):
        left_indicators = BaseCensoredLifetime._check_indicators(
            values, left_indicators
        )
        self.left_index = np.where(left_indicators)[0]

    def set_right_index(
        self,
        values: np.ndarray,
        right_indicators: np.ndarray = np.array([], dtype=int),
        **kwargs,
    ):
        right_indicators = BaseCensoredLifetime._check_indicators(
            values, right_indicators
        )
        self.right_index = np.where(right_indicators)[0]

    def set_interval_index(self, **kwargs):
        pass

    def set_regular_index(self, **kwargs):
        self.regular_index = np.delete(
            self.regular_index,
            list(set(self.left_index).union(set(self.right_index))),
        )

    def set_left_values(self, values: np.ndarray, **kwargs):
        self.left_values = values[self.left_index]

    def set_right_values(self, values: np.ndarray, **kwargs):
        self.right_values = values[self.right_index]

    def set_interval_values(self, values: np.ndarray, **kwargs):
        self.interval_values = values[self.interval_index]

    def set_regular_values(self, values: np.ndarray, **kwargs):
        self.regular_values = values[self.regular_index]


class AdvancedCensoredLifetime(LifetimeFormat):
    def __init__(self, values=np.ndarray):
        super().__init__(values)

    def set_left_index(
        self,
        values: np.ndarray,
        **kwargs,
    ):
        self.left_index = np.where(values[:, 0] == 0.0)[0]

    def set_right_index(
        self,
        values: np.ndarray,
        **kwargs,
    ):
        self.right_index = np.where(values[:, 1] == np.inf)[0]

    def set_interval_index(
        self,
        values: np.ndarray,
        **kwargs,
    ):
        self.interval_index = np.where(
            np.logical_and(
                np.logical_and(
                    values[:, 0] > 0,
                    values[:, 1] < np.inf,
                ),
                np.not_equal(values[:, 0], values[:, 1]),
            )
        )[0]

    def set_regular_index(self, **kwargs):
        self.regular_index = np.delete(
            self.regular_index,
            list(
                set(self.left_index)
                .union(set(self.right_index))
                .union(self.interval_index)
            ),
        )

    def set_left_values(self, values: np.ndarray, **kwargs):
        self.left_values = values[self.left_index, :][:, 1]

    def set_right_values(self, values: np.ndarray, **kwargs):
        self.right_values = values[self.right_index, :][:, 0]

    def set_interval_values(self, values: np.ndarray, **kwargs):
        self.interval_values = values[self.interval_index, :]

    def set_regular_values(self, values: np.ndarray, **kwargs):
        assert (
            values[self.regular_index][:, 0] == values[self.regular_index][:, 1]
        ).all()
        self.regular_values = values[self.regular_index][:, 0]


class Truncation(LifetimeFormat):
    def __init__(self, values=np.ndarray):
        super().__init__(values)

    def set_left_index(
        self,
        values: np.ndarray,
        entry: np.ndarray,
        **kwargs,
    ):
        if entry.size != 0:
            assert values.size == entry.size and values.shape == entry.shape
            if np.any(entry < 0):
                raise ValueError("entry values must be positive")
            if np.any(values <= entry):
                raise ValueError("entry must be strictly lower than the lifetimes")
        self.left_index = np.where(entry > 0)[0]

    def set_right_index(
        self,
        values: np.ndarray,
        departure: np.ndarray,
        **kwargs,
    ):
        if departure.size != 0:
            assert values.size == departure.size and values.shape == departure.shape
            if np.any(departure < 0):
                raise ValueError("entry values must be positive")
            if np.any(values > departure):
                raise ValueError("departure must be higher or equal to lifetimes")
        self.right_index = np.where(departure > 0)[0]

    def set_interval_index(self, entry: np.ndarray, departure: np.ndarray, **kwargs):
        if entry.size != 0 and departure.size != 0:
            self.interval_index = np.where(np.logical_and(entry > 0, departure > 0))[0]

    def set_regular_index(self, **kwargs):
        self.regular_index = np.delete(
            self.regular_index,
            list(
                set(self.left_index)
                .union(set(self.right_index))
                .union(self.interval_index)
            ),
        )

    def set_left_values(self, entry: np.ndarray, **kwargs):
        self.left_values = entry[self.left_index]

    def set_right_values(self, departure: np.ndarray, **kwargs):
        self.right_values = departure[self.right_index]

    def set_interval_values(self, entry: np.ndarray, departure: np.ndarray, **kwargs):
        self.interval_values = np.concatenate(
            (
                entry[self.interval_index][:, None],
                departure[self.interval_index][:, None],
            ),
            axis=1,
        )

    def set_regular_values(self, values: np.ndarray, **kwargs):
        self.regular_values = values[self.regular_index]
