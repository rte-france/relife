from abc import ABC, abstractmethod

import numpy as np


class LifetimeDecoder(ABC):
    def __init__(self, values: np.ndarray):
        self.values = values

    @abstractmethod
    def get_left_index(
        self,
    ):
        pass

    @abstractmethod
    def get_right_index(
        self,
    ):
        pass

    @abstractmethod
    def get_interval_index(
        self,
    ):
        pass

    @abstractmethod
    def get_regular_index(
        self,
    ):
        pass

    @abstractmethod
    def get_left_values(
        self,
    ):
        pass

    @abstractmethod
    def get_right_values(
        self,
    ):
        pass

    @abstractmethod
    def get_interval_values(
        self,
    ):
        pass

    @abstractmethod
    def get_regular_values(
        self,
    ):
        pass


class BaseCensoredLifetime(LifetimeDecoder):
    def __init__(
        self,
        values=np.ndarray,
        left_indicators: np.ndarray = np.array([], dtype=int),
        right_indicators: np.ndarray = np.array([], dtype=int),
    ):
        super().__init__(values)
        self.left_indicators = self._check_indicators(left_indicators)
        self.right_indicators = self._check_indicators(right_indicators)

    def _check_indicators(self, indicators: np.ndarray):
        if type(indicators) == np.ndarray:
            if indicators.size != 0:
                assert len(indicators.shape) == 1, "indicators must be 1d array"
                assert len(indicators) == len(
                    self.values
                ), "indicators must have the same length as lifetime values"
                if indicators.dtype != np.bool_:
                    indicators = indicators.astype(bool)
        else:
            ValueError("indicators must be np.ndarray")
        return indicators

    def get_left_index(
        self,
    ):
        return np.where(self.left_indicators)[0]

    def get_right_index(
        self,
    ):
        return np.where(self.right_indicators)[0]

    def get_interval_index(self):
        return np.where(np.zeros(len(self.values), dtype=int))[0]

    def get_regular_index(
        self,
    ):
        return np.delete(
            np.arange(0, len(self.values)),
            list(set(self.get_left_index()).union(set(self.get_right_index()))),
        )

    def get_left_values(self):
        return self.values[self.get_left_index()]

    def get_right_values(self):
        return self.values[self.get_right_index()]

    def get_interval_values(self):
        return self.values[self.get_interval_index()]

    def get_regular_values(self):
        return self.values[self.get_regular_index()]


class AdvancedCensoredLifetime(LifetimeDecoder):
    def __init__(self, values=np.ndarray):
        super().__init__(values)

    def get_left_index(
        self,
    ):
        return np.where(self.values[:, 0] == 0.0)[0]

    def get_right_index(
        self,
    ):
        return np.where(self.values[:, 1] == np.inf)[0]

    def get_interval_index(
        self,
    ):
        return np.where(
            np.logical_and(
                np.logical_and(
                    self.values[:, 0] > 0,
                    self.values[:, 1] < np.inf,
                ),
                np.not_equal(self.values[:, 0], self.values[:, 1]),
            )
        )[0]

    def get_regular_index(self):
        return np.delete(
            np.arange(0, len(self.values)),
            list(
                set(self.get_left_index())
                .union(set(self.get_right_index()))
                .union(self.get_interval_index())
            ),
        )

    def get_left_values(self):
        return self.values[self.get_left_index(), :][:, 1]

    def get_right_values(self):
        return self.values[self.get_right_index(), :][:, 0]

    def get_interval_values(self):
        return self.values[self.get_interval_index(), :]

    def get_regular_values(self):
        assert (
            self.values[self.get_regular_index()][:, 0]
            == self.values[self.get_regular_index()][:, 1]
        ).all()
        return self.values[self.get_regular_index()][:, 0]


class Truncation(LifetimeDecoder):
    def __init__(
        self,
        values=np.ndarray,
        entry: np.ndarray = np.array([], dtype=float),
        departure: np.ndarray = np.array([], dtype=float),
    ):
        super().__init__(values)
        self.entry = entry
        self.departure = departure

        if self.entry.size != 0:
            assert (
                self.values.size == self.entry.size
                and self.values.shape == self.entry.shape
            )
            if np.any(self.entry < 0):
                raise ValueError("entry values must be positive")
            if np.any(self.values <= self.entry):
                raise ValueError("entry must be strictly lower than the lifetimes")

        if self.departure.size != 0:
            assert (
                self.values.size == self.departure.size
                and self.values.shape == self.departure.shape
            )
            if np.any(self.departure < 0):
                raise ValueError("entry values must be positive")
            if np.any(self.values > self.departure):
                raise ValueError("departure must be higher or equal to lifetimes")

    def get_left_index(
        self,
    ):
        return np.where(self.entry > 0)[0]

    def get_right_index(
        self,
    ):

        return np.where(self.departure > 0)[0]

    def get_interval_index(
        self,
    ):
        if self.entry.size != 0 and self.departure.size != 0:
            return np.where(np.logical_and(self.entry > 0, self.departure > 0))[0]
        else:
            return np.where(np.zeros(len(self.values), dtype=int))[0]

    def get_regular_index(
        self,
    ):
        return np.delete(
            np.arange(0, len(self.values)),
            list(
                set(self.get_left_index())
                .union(set(self.get_right_index()))
                .union(self.get_interval_index())
            ),
        )

    def get_left_values(
        self,
    ):
        return self.entry[self.get_left_index()]

    def get_right_values(
        self,
    ):
        return self.departure[self.get_right_index()]

    def get_interval_values(
        self,
    ):
        return np.concatenate(
            (
                self.entry[self.get_interval_index()][:, None],
                self.departure[self.get_interval_index()][:, None],
            ),
            axis=1,
        )

    def get_regular_values(
        self,
    ):
        return self.values[self.get_regular_index()]


# factory
def censoredlifetimes_decoder(
    lifetime_values: np.ndarray,
    left_indicators: np.ndarray = np.array([], dtype=bool),
    right_indicators: np.ndarray = np.array([], dtype=bool),
):
    if len(lifetime_values.shape) == 1:
        constructor = BaseCensoredLifetime(
            lifetime_values, left_indicators, right_indicators
        )
    elif len(lifetime_values.shape) == 2:
        constructor = AdvancedCensoredLifetime(lifetime_values)
    else:
        return ValueError("lifetimes values must be 1d or 2d array")
    return constructor


# factory
def truncations_decoder(
    lifetime_values: np.ndarray,
    entry: np.ndarray = np.array([], dtype=float),
    departure: np.ndarray = np.array([], dtype=float),
):
    constructor = Truncation(lifetime_values, entry, departure)
    return constructor
