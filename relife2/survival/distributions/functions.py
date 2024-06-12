"""
This module defines probability functions used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from typing import Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv, polygamma

from relife2.survival.data import ObservedLifetimes
from relife2.survival.distributions.types import DistributionFunctions
from relife2.survival.integrations import shifted_laguerre

FloatArray = NDArray[np.float64]

# pylint: disable=no-member


class ExponentialFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def __init__(self, rate: Optional[float] = None):
        super().__init__(rate=rate)

    def hf(self, time: FloatArray) -> FloatArray:
        return self.params.rate * np.ones_like(time)

    def chf(self, time: FloatArray) -> FloatArray:
        return self.params.rate * time

    def mean(self) -> float:
        return 1 / self.params.rate

    def var(self) -> float:
        return 1 / self.params.rate**2

    def mrl(self, time: FloatArray) -> FloatArray:
        return 1 / self.params.rate * np.ones_like(time)

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return cumulative_hazard_rate / self.params.rate

    def jac_hf(self, time: FloatArray) -> FloatArray:
        return np.ones((time.size, 1))

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.ones((time.size, 1)) * time

    def dhf(self, time: FloatArray) -> Union[float, FloatArray]:
        return np.zeros_like(time)


class WeibullFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    def hf(self, time: FloatArray) -> FloatArray:
        return (
            self.params.shape
            * self.params.rate
            * (self.params.rate * time) ** (self.params.shape - 1)
        )

    def chf(self, time: FloatArray) -> FloatArray:
        return (self.params.rate * time) ** self.params.shape

    def mean(self) -> float:
        return gamma(1 + 1 / self.params.shape) / self.params.rate

    def var(self) -> float:
        return gamma(1 + 2 / self.params.shape) / self.params.rate**2 - self.mean() ** 2

    def mrl(self, time: FloatArray) -> FloatArray:
        return (
            gamma(1 / self.params.shape)
            / (self.params.rate * self.params.shape * self.sf(time))
            * gammaincc(
                1 / self.params.shape,
                (self.params.rate * time) ** self.params.shape,
            )
        )

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return cumulative_hazard_rate ** (1 / self.params.shape) / self.params.rate

    def jac_hf(self, time: FloatArray) -> FloatArray:

        return np.column_stack(
            (
                self.params.rate
                * (self.params.rate * time) ** (self.params.shape - 1)
                * (1 + self.params.shape * np.log(self.params.rate * time)),
                self.params.shape**2
                * (self.params.rate * time) ** (self.params.shape - 1),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                np.log(self.params.rate * time)
                * (self.params.rate * time) ** self.params.shape,
                self.params.shape
                * time
                * (self.params.rate * time) ** (self.params.shape - 1),
            )
        )

    def dhf(self, time: FloatArray) -> Union[float, FloatArray]:
        return (
            self.params.shape
            * (self.params.shape - 1)
            * self.params.rate**2
            * (self.params.rate * time) ** (self.params.shape - 2)
        )


class GompertzFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    def initial_params(self, lifetimes: ObservedLifetimes) -> FloatArray:
        param0 = np.empty(self.params.size, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(lifetimes.rlc.values))
        shape = np.exp(-rate * np.mean(lifetimes.rlc.values))
        param0[0] = shape
        param0[1] = rate

        return param0

    def hf(self, time: FloatArray) -> FloatArray:
        return self.params.shape * self.params.rate * np.exp(self.params.rate * time)

    def chf(self, time: FloatArray) -> FloatArray:
        return self.params.shape * np.expm1(self.params.rate * time)

    def mean(self) -> float:
        return np.exp(self.params.shape) * exp1(self.params.shape) / self.params.rate

    def var(self) -> Any:
        return polygamma(1, 1) / self.params.rate**2

    def mrl(self, time: FloatArray) -> FloatArray:
        z = self.params.shape * np.exp(self.params.rate * time)
        return np.exp(z) * exp1(z) / self.params.rate

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return (
            1 / self.params.rate * np.log1p(cumulative_hazard_rate / self.params.shape)
        )

    def jac_hf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                self.params.rate * np.exp(self.params.rate * time),
                self.params.shape
                * np.exp(self.params.rate * time)
                * (1 + self.params.rate * time),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                np.expm1(self.params.rate * time),
                self.params.shape * time * np.exp(self.params.rate * time),
            )
        )

    def dhf(self, time: FloatArray) -> Union[float, FloatArray]:
        return self.params.shape * self.params.rate**2 * np.exp(self.params.rate * time)


class GammaFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    def _uppergamma(self, x: FloatArray) -> FloatArray:
        return gammaincc(self.params.shape, x) * gamma(self.params.shape)

    def _jac_uppergamma_shape(self, x: FloatArray) -> FloatArray:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (self.params.shape - 1),
            x,
            ndim=np.ndim(x),
        )

    def hf(self, time: FloatArray) -> FloatArray:
        x = self.params.rate * time
        return (
            self.params.rate
            * x ** (self.params.shape - 1)
            * np.exp(-x)
            / self._uppergamma(x)
        )

    def chf(self, time: FloatArray) -> FloatArray:
        x = self.params.rate * time
        return np.log(gamma(self.params.shape)) - np.log(self._uppergamma(x))

    def mean(self) -> float:
        return self.params.shape / self.params.rate

    def var(self) -> float:
        return self.params.shape / (self.params.rate**2)

    def mrl(self, time: FloatArray) -> FloatArray:
        return super().mrl(time)

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return (
            1
            / self.params.rate
            * gammainccinv(self.params.shape, np.exp(-cumulative_hazard_rate))
        )

    def jac_hf(self, time: FloatArray) -> FloatArray:

        x = self.params.rate * time
        return (
            x ** (self.params.shape - 1)
            * np.exp(-x)
            / self._uppergamma(x) ** 2
            * np.column_stack(
                (
                    self.params.rate * np.log(x) * self._uppergamma(x)
                    - self.params.rate * self._jac_uppergamma_shape(x),
                    (self.params.shape - x) * self._uppergamma(x)
                    + x**self.params.shape * np.exp(-x),
                )
            )
        )

    def jac_chf(
        self,
        time: FloatArray,
    ) -> FloatArray:
        x = self.params.rate * time
        return np.column_stack(
            (
                digamma(self.params.shape)
                - self._jac_uppergamma_shape(x) / self._uppergamma(x),
                x ** (self.params.shape - 1) * time * np.exp(-x) / self._uppergamma(x),
            )
        )

    def dhf(self, time: FloatArray) -> Union[float, FloatArray]:
        return self.hf(time) * (
            (self.params.shape - 1) / time - self.params.rate + self.hf(time)
        )


class LogLogisticFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(shape=shape, rate=rate)

    def hf(self, time: FloatArray) -> FloatArray:
        x = self.params.rate * time
        return (
            self.params.shape
            * self.params.rate
            * x ** (self.params.shape - 1)
            / (1 + x**self.params.shape)
        )

    def chf(self, time: FloatArray) -> FloatArray:
        x = self.params.rate * time
        return np.log(1 + x**self.params.shape)

    def mean(self) -> float:
        b = np.pi / self.params.shape
        if self.params.shape <= 1:
            raise ValueError(
                f"Expectancy only defined for c > 1: c = {self.params.shape}"
            )
        return b / (self.params.rate * np.sin(b))

    def var(self) -> float:
        b = np.pi / self.params.shape
        if self.params.shape <= 2:
            raise ValueError(
                f"Variance only defined for c > 2: c = {self.params.shape}"
            )
        return (1 / self.params.rate**2) * (
            2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2)
        )

    def mrl(self, time: FloatArray) -> FloatArray:
        return super().mrl(time)

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return (
            (np.exp(cumulative_hazard_rate) - 1) ** (1 / self.params.shape)
        ) / self.params.rate

    def jac_hf(self, time: FloatArray) -> FloatArray:
        x = self.params.rate * time
        return np.column_stack(
            (
                (
                    self.params.rate
                    * x ** (self.params.shape - 1)
                    / (1 + x**self.params.shape) ** 2
                )
                * (
                    1
                    + x**self.params.shape
                    + self.params.shape * np.log(self.params.rate * time)
                ),
                (
                    self.params.rate
                    * x ** (self.params.shape - 1)
                    / (1 + x**self.params.shape) ** 2
                )
                * (self.params.shape**2 / self.params.rate),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        x = self.params.rate * time
        return np.column_stack(
            (
                (x**self.params.shape / (1 + x**self.params.shape))
                * np.log(self.params.rate * time),
                (x**self.params.shape / (1 + x**self.params.shape))
                * (self.params.shape / self.params.rate),
            )
        )

    def dhf(self, time: FloatArray) -> Union[float, FloatArray]:
        x = self.params.rate * time
        return (
            self.params.shape
            * self.params.rate**2
            * x ** (self.params.shape - 2)
            * (self.params.shape - 1 - x**self.params.shape)
            / (1 + x**self.params.shape) ** 2
        )
