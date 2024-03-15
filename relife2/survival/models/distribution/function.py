import numpy as np
from scipy.special import exp1, gamma, gammaincc

from ..interface.distribution import ParametricDistFunction


class ExponentialDistFunction(ParametricDistFunction):
    def __init__(self, param_names=["rate"]):
        super().__init__(param_names=param_names)

    # relife/distribution.Exponential
    # mandatory
    def hf(self, time: np.ndarray) -> np.ndarray:
        """Hazard function

        Args:
            time (np.ndarray): time

        Returns:
            np.ndarray: hazard function values
        """
        # rate = self.params[0]
        return self.params.rate * np.ones_like(time)

    # relife/distribution.Exponential
    # mandatory
    def chf(self, time: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return self.params.rate * time

    # relife/distribution.Exponential
    # mandatory
    def mean(self) -> float:
        # rate = self.params[0]
        return 1 / self.params.rate

    # relife/distribution.Exponential
    # mandatory
    def var(self) -> float:
        # rate = self.params[0]
        return 1 / self.params.rate**2

    # relife/distribution.Exponential
    # mandatory
    def mrl(self, time: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return 1 / self.params.rate * np.ones_like(time)

    # relife/distribution.Exponential /!\ dependant of _ichf (why : carry fitted params and params)
    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return cumulative_hazard_rate / self.params.rate


class WeibullDistFunction(ParametricDistFunction):
    def __init__(self, param_names=["c", "rate"]):
        super().__init__(param_names=param_names)

    def hf(self, time: np.ndarray) -> np.ndarray:
        return (
            self.params.c
            * self.params.rate
            * (self.params.rate * time) ** (self.params.c - 1)
        )

    def chf(self, time: np.ndarray) -> np.ndarray:
        return (self.params.rate * time) ** self.params.c

    def mean(self) -> float:
        return gamma(1 + 1 / self.params.c) / self.params.rate

    def var(self) -> float:
        return (
            gamma(1 + 2 / self.params.c) / self.params.rate**2
            - self.mean() ** 2
        )

    def mrl(self, time: np.ndarray) -> np.ndarray:
        return (
            gamma(1 / self.params.c)
            / (self.params.rate * self.params.c * self.sf(time))
            * gammaincc(
                1 / self.params.c,
                (self.params.rate * time) ** self.params.c,
            )
        )

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return cumulative_hazard_rate ** (1 / self.params.c) / self.params.rate


class GompertzDistFunction(ParametricDistFunction):
    def __init__(self, param_names=["c", "rate"]):
        super().__init__(param_names=param_names)

    def hf(self, time: np.ndarray) -> np.ndarray:
        return (
            self.params.c * self.params.rate * np.exp(self.params.rate * time)
        )

    def chf(self, time: np.ndarray) -> np.ndarray:
        return self.params.c * np.expm1(self.params.rate * time)

    def mean(self) -> float:
        return np.exp(self.params.c) * exp1(self.params.c) / self.params.rate

    def var(self) -> float:
        pass

    def mrl(self, time: np.ndarray) -> np.ndarray:
        z = self.params.c * np.exp(self.params.rate * time)
        return np.exp(z) * exp1(z) / self.params.rate

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return (
            1
            / self.params.rate
            * np.log1p(cumulative_hazard_rate / self.params.c)
        )