from abc import abstractmethod
from typing import List

import numpy as np
from scipy.special import exp1, gamma, gammaincc, gammainccinv

from ..backbone import ProbabilityFunctions


class DistFunctions(ProbabilityFunctions):
    def __init__(self, nb_params: int, param_names: List[str] = None):
        super().__init__(nb_params, param_names)

    @abstractmethod
    def ichf(self, cumulative_hazard_rate: np.ndarray):
        """only mandatory for DistFunctions as exact expression is known"""
        pass

    # relife/model.AbsolutelyContinuousLifetimeModel /!\ dependant of ichf and _ichf
    # /!\ mathematically : -np.log(probability) = cumulative_hazard_rate
    def isf(self, probability: np.ndarray) -> np.ndarray:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)


class ExponentialFunctions(DistFunctions):
    def __init__(self):
        super().__init__(1, ["rate"])

    # relife/distribution.Exponential
    # mandatory
    def hf(self, time: np.ndarray) -> np.ndarray:
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


class WeibullFunctions(DistFunctions):
    def __init__(self):
        super().__init__(2, ["c", "rate"])

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


class GompertzFunctions(DistFunctions):
    def __init__(self):
        super().__init__(2, ["c", "rate"])

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


class GammaFunctions(DistFunctions):
    def __init__(self):
        super().__init__(2, ["c", "rate"])

    def _uppergamma(self, x: np.ndarray) -> np.ndarray:
        return gammaincc(self.params.c, x) * gamma(self.params.c)

    def hf(self, time: np.ndarray) -> np.ndarray:
        x = self.params.rate * time
        return (
            self.params.rate
            * x ** (self.params.c - 1)
            * np.exp(-x)
            / self._uppergamma(x)
        )

    def chf(self, time: np.ndarray) -> np.ndarray:
        x = self.params.rate * time
        return np.log(gamma(self.params.c)) - np.log(self._uppergamma(x))

    def mean(self) -> float:
        return self.params.c / self.params.rate

    def var(self) -> float:
        return self.params.c / (self.params.rate**2)

    # def mrl(self, time: np.ndarray) -> np.ndarray:
    #     pass

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return (
            1
            / self.params.rate
            * gammainccinv(self.params.c, np.exp(-cumulative_hazard_rate))
        )


class LogLogisticFunctions(DistFunctions):
    def __init__(self):
        super().__init__(2, ["c", "rate"])

    def hf(self, time: np.ndarray) -> np.ndarray:
        x = self.params.rate * time
        return (
            self.params.c
            * self.params.rate
            * x ** (self.params.c - 1)
            / (1 + x**self.params.c)
        )

    def chf(self, time: np.ndarray) -> np.ndarray:
        x = self.params.rate * time
        return np.log(1 + x**self.params.c)

    def mean(self) -> np.ndarray:
        b = np.pi / self.params.c
        if self.params.c > 1:
            return b / (self.params.rate * np.sin(b))
        else:
            raise ValueError(
                f"Expectancy only defined for c > 1: c = {self.params.c}"
            )

    def var(self) -> np.ndarray:
        b = np.pi / self.params.c
        if self.params.c > 2:
            return (1 / self.params.rate**2) * (
                2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2)
            )
        else:
            raise ValueError(
                f"Variance only defined for c > 2: c = {self.params.c}"
            )

    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        return (
            (np.exp(cumulative_hazard_rate) - 1) ** (1 / self.params.c)
        ) / self.params.rate


class MinimumDistFunctions(DistFunctions):
    def __init__(self, baseline: DistFunctions, n: np.ndarray):
        self.baseline = baseline
        self.n = n
        super().__init__(
            baseline.params.nb_params, baseline.params.param_names
        )

    @property
    def _default_hess_scheme(self) -> str:
        return self.baseline._default_hess_scheme

    def _set_params(self, params: np.ndarray) -> None:
        self.baseline._set_params(params)

    def _chf(
        self,
        params: np.ndarray,
        t: np.ndarray,
        n: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return n * self.baseline._chf(params, t, *args)

    def _hf(
        self,
        params: np.ndarray,
        t: np.ndarray,
        n: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return n * self.baseline._hf(params, t, *args)

    def _dhf(
        self,
        params: np.ndarray,
        t: np.ndarray,
        n: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return n * self.baseline._dhf(params, t, *args)

    def _jac_chf(
        self,
        params: np.ndarray,
        t: np.ndarray,
        n: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return n * self.baseline._jac_chf(params, t, *args)

    def _jac_hf(
        self,
        params: np.ndarray,
        t: np.ndarray,
        n: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return n * self.baseline._jac_hf(params, t, *args)

    def _ichf(
        self,
        params: np.ndarray,
        v: np.ndarray,
        n: np.ndarray,
        *args: np.ndarray,
    ) -> np.ndarray:
        return self.baseline._ichf(params, v / n, *args)
