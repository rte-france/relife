from abc import ABC
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv, polygamma

from relife2.data import LifetimeData
from relife2.fiability.model import ParametricLifetimeModel
from relife2.maths.integration import shifted_laguerre


# Ts type var is a zero long tuple (see https://github.com/python/mypy/issues/16199)
# note : Tuple[()] behaves differently (to follow)
# no args are required
class Distribution(ParametricLifetimeModel[()], ABC):
    def sf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return super().sf(time)

    def isf(self, probability: NDArray[np.float64]) -> NDArray[np.float64]:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)

    def cdf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return super().cdf(time)

    def pdf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return super().pdf(time)

    def ppf(self, probability: NDArray[np.float64]) -> NDArray[np.float64]:
        return super().ppf(probability)

    def rvs(self, *, size: Optional[int] = 1, seed: Optional[int] = None):
        return super().rvs(size=size, seed=seed)

    def median(self):
        return super().median()

    def init_params(self, lifetime_data: LifetimeData) -> None:
        param0 = np.ones(self.nb_params)
        param0[-1] = 1 / np.median(lifetime_data.rlc.values)
        self.params = param0

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    # @property
    # def support_lower_bound(self):
    #     return 0.0
    #
    # @property
    # def support_upper_bound(self):
    #     return np.inf


class Exponential(Distribution):
    def __init__(self, rate: Optional[float] = None):
        super().__init__()
        self.new_params(rate=rate)

    def hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Args:
            time (NDArray[np.float64]):
        Returns:

        """
        return self.rate * np.ones_like(time)

    def chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.rate * time

    def mean(self) -> NDArray[np.float64]:
        return np.array(1 / self.rate)

    def var(self) -> NDArray[np.float64]:
        return np.array(1 / self.rate**2)

    def mrl(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / self.rate * np.ones_like(time)

    def ichf(self, cumulative_hazard_rate: NDArray[np.float64]) -> NDArray[np.float64]:
        return cumulative_hazard_rate / self.rate

    def jac_hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones((time.size, 1))

    def jac_chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ones((time.size, 1)) * time

    def dhf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.zeros_like(time)


class Weibull(Distribution):
    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.shape * self.rate * (self.rate * time) ** (self.shape - 1)

    def chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return (self.rate * time) ** self.shape

    def mean(self) -> NDArray[np.float64]:
        return np.array(gamma(1 + 1 / self.shape) / self.rate)

    def var(self) -> NDArray[np.float64]:
        return np.array(gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2)

    def mrl(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    def ichf(self, cumulative_hazard_rate: NDArray[np.float64]) -> NDArray[np.float64]:
        return cumulative_hazard_rate ** (1 / self.shape) / self.rate

    def jac_hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:

        return np.column_stack(
            (
                self.rate
                * (self.rate * time) ** (self.shape - 1)
                * (1 + self.shape * np.log(self.rate * time)),
                self.shape**2 * (self.rate * time) ** (self.shape - 1),
            )
        )

    def jac_chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            (
                np.log(self.rate * time) * (self.rate * time) ** self.shape,
                self.shape * time * (self.rate * time) ** (self.shape - 1),
            )
        )

    def dhf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return (
            self.shape
            * (self.shape - 1)
            * self.rate**2
            * (self.rate * time) ** (self.shape - 2)
        )


class Gompertz(Distribution):
    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def init_params(self, lifetime_data: LifetimeData) -> None:
        param0 = np.empty(self.nb_params, dtype=float)
        rate = np.pi / (np.sqrt(6) * np.std(lifetime_data.rlc.values))
        shape = np.exp(-rate * np.mean(lifetime_data.rlc.values))
        param0[0] = shape
        param0[1] = rate
        self.params = param0

    def hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.shape * self.rate * np.exp(self.rate * time)

    def chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.shape * np.expm1(self.rate * time)

    def mean(self) -> NDArray[np.float64]:
        return np.array(np.exp(self.shape) * exp1(self.shape) / self.rate)

    def var(self) -> NDArray[np.float64]:
        return np.array(polygamma(1, 1).item() / self.rate**2)

    def mrl(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    def ichf(self, cumulative_hazard_rate: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    def jac_hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            (
                self.rate * np.exp(self.rate * time),
                self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
            )
        )

    def jac_chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.column_stack(
            (
                np.expm1(self.rate * time),
                self.shape * time * np.exp(self.rate * time),
            )
        )

    def dhf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.shape * self.rate**2 * np.exp(self.rate * time)


class Gamma(Distribution):
    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def _uppergamma(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (self.shape - 1),
            x,
            ndim=np.ndim(x),
        )

    def hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    def chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    def mean(self) -> NDArray[np.float64]:
        return np.array(self.shape / self.rate)

    def var(self) -> NDArray[np.float64]:
        return np.array(self.shape / (self.rate**2))

    def ichf(self, cumulative_hazard_rate: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1 / self.rate * gammainccinv(self.shape, np.exp(-cumulative_hazard_rate))

    def jac_hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:

        x = self.rate * time
        return (
            x ** (self.shape - 1)
            * np.exp(-x)
            / self._uppergamma(x) ** 2
            * np.column_stack(
                (
                    self.rate * np.log(x) * self._uppergamma(x)
                    - self.rate * self._jac_uppergamma_shape(x),
                    (self.shape - x) * self._uppergamma(x) + x**self.shape * np.exp(-x),
                )
            )
        )

    def jac_chf(
        self,
        time: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        x = self.rate * time
        return np.column_stack(
            (
                digamma(self.shape)
                - self._jac_uppergamma_shape(x) / self._uppergamma(x),
                x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x),
            )
        )

    def dhf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))

    def mrl(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return super().mrl(time)


class LogLogistic(Distribution):
    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    def chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return np.array(np.log(1 + x**self.shape))

    def mean(self) -> NDArray[np.float64]:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(
                f"Expectancy only defined for shape > 1: shape = {self.shape}"
            )
        return np.array(b / (self.rate * np.sin(b)))

    def var(self) -> NDArray[np.float64]:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(
                f"Variance only defined for shape > 2: shape = {self.shape}"
            )
        return np.array(
            (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))
        )

    def ichf(self, cumulative_hazard_rate: NDArray[np.float64]) -> NDArray[np.float64]:
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    def jac_hf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return np.column_stack(
            (
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (self.shape**2 / self.rate),
            )
        )

    def jac_chf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return np.column_stack(
            (
                (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
                (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
            )
        )

    def dhf(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        x = self.rate * time
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    def mrl(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return super().mrl(time)
