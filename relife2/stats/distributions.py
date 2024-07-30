"""
This module defines probability functions used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import Bounds
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv, polygamma

from relife2.stats.functions import ParametricLifetimeFunctions
from relife2.utils.integrations import shifted_laguerre
from relife2.utils.types import FloatArray

# pylint: disable=no-member


class DistributionFunctions(ParametricLifetimeFunctions, ABC):
    """
    Object that computes every probability functions of a distribution model
    """

    def init_params(self, *args: Any) -> FloatArray:
        param0 = np.ones(self.nb_params)
        param0[-1] = 1 / np.median(args[0].values)
        return param0

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLA
        """
        return 0.0

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLA
        """
        return np.inf

    @abstractmethod
    def jac_hf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def jac_chf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def dhf(self, time: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def ichf(
        self,
        cumulative_hazard_rate: FloatArray,
    ) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
        Returns:
            FloatArray: BLABLABLABLA
        """

    def isf(self, probability: FloatArray) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA
        Returns:
            FloatArray: BLABLABLABLA
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)


class ExponentialFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def hf(self, time: FloatArray) -> FloatArray:
        return self.rate * np.ones_like(time)

    def chf(self, time: FloatArray) -> FloatArray:
        return self.rate * time

    def mean(self) -> FloatArray:
        return np.array(1 / self.rate)

    def var(self) -> FloatArray:
        return np.array(1 / self.rate**2)

    def mrl(self, time: FloatArray) -> FloatArray:
        return 1 / self.rate * np.ones_like(time)

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return cumulative_hazard_rate / self.rate

    def jac_hf(self, time: FloatArray) -> FloatArray:
        return np.ones((time.size, 1))

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.ones((time.size, 1)) * time

    def dhf(self, time: FloatArray) -> FloatArray:
        return np.zeros_like(time)


class WeibullFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def hf(self, time: FloatArray) -> FloatArray:
        return self.shape * self.rate * (self.rate * time) ** (self.shape - 1)

    def chf(self, time: FloatArray) -> FloatArray:
        return (self.rate * time) ** self.shape

    def mean(self) -> FloatArray:
        return np.array(gamma(1 + 1 / self.shape) / self.rate)

    def var(self) -> FloatArray:
        return np.array(gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2)

    def mrl(self, time: FloatArray) -> FloatArray:
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return cumulative_hazard_rate ** (1 / self.shape) / self.rate

    def jac_hf(self, time: FloatArray) -> FloatArray:

        return np.column_stack(
            (
                self.rate
                * (self.rate * time) ** (self.shape - 1)
                * (1 + self.shape * np.log(self.rate * time)),
                self.shape**2 * (self.rate * time) ** (self.shape - 1),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                np.log(self.rate * time) * (self.rate * time) ** self.shape,
                self.shape * time * (self.rate * time) ** (self.shape - 1),
            )
        )

    def dhf(self, time: FloatArray) -> FloatArray:
        return (
            self.shape
            * (self.shape - 1)
            * self.rate**2
            * (self.rate * time) ** (self.shape - 2)
        )


class GompertzFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def init_params(self, *args: Any) -> FloatArray:
        param0 = np.empty(self.nb_params, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(args[0].values))
        shape = np.exp(-rate * np.mean(args[0].values))
        param0[0] = shape
        param0[1] = rate

        return param0

    def hf(self, time: FloatArray) -> FloatArray:
        return self.shape * self.rate * np.exp(self.rate * time)

    def chf(self, time: FloatArray) -> FloatArray:
        return self.shape * np.expm1(self.rate * time)

    def mean(self) -> FloatArray:
        return np.array(np.exp(self.shape) * exp1(self.shape) / self.rate)

    def var(self) -> FloatArray:
        return np.array(polygamma(1, 1).item() / self.rate**2)

    def mrl(self, time: FloatArray) -> FloatArray:
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    def jac_hf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                self.rate * np.exp(self.rate * time),
                self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        return np.column_stack(
            (
                np.expm1(self.rate * time),
                self.shape * time * np.exp(self.rate * time),
            )
        )

    def dhf(self, time: FloatArray) -> FloatArray:
        return self.shape * self.rate**2 * np.exp(self.rate * time)


class GammaFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def _uppergamma(self, x: FloatArray) -> FloatArray:
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: FloatArray) -> FloatArray:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (self.shape - 1),
            x,
            ndim=np.ndim(x),
        )

    def hf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    def chf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    def mean(self) -> FloatArray:
        return np.array(self.shape / self.rate)

    def var(self) -> FloatArray:
        return np.array(self.shape / (self.rate**2))

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return 1 / self.rate * gammainccinv(self.shape, np.exp(-cumulative_hazard_rate))

    def jac_hf(self, time: FloatArray) -> FloatArray:

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
        time: FloatArray,
    ) -> FloatArray:
        x = self.rate * time
        return np.column_stack(
            (
                digamma(self.shape)
                - self._jac_uppergamma_shape(x) / self._uppergamma(x),
                x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x),
            )
        )

    def dhf(self, time: FloatArray) -> FloatArray:
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))


class LogLogisticFunctions(DistributionFunctions):
    """
    BLABLABLABLA
    """

    def hf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    def chf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return np.array(np.log(1 + x**self.shape))

    def mean(self) -> FloatArray:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(f"Expectancy only defined for c > 1: c = {self.shape}")
        return np.array(b / (self.rate * np.sin(b)))

    def var(self) -> FloatArray:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(f"Variance only defined for c > 2: c = {self.shape}")
        return np.array(
            (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))
        )

    def ichf(self, cumulative_hazard_rate: FloatArray) -> FloatArray:
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    def jac_hf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return np.column_stack(
            (
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (self.shape**2 / self.rate),
            )
        )

    def jac_chf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return np.column_stack(
            (
                (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
                (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
            )
        )

    def dhf(self, time: FloatArray) -> FloatArray:
        x = self.rate * time
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )
