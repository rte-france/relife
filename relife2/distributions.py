"""
This module defines probability functions used in distributions

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from scipy.optimize import Bounds
from scipy.special import (
    digamma,
    exp1,
    expi,
    gamma,
    gammainc,
    gammaincc,
    gammainccinv,
    lambertw,
    loggamma,
    polygamma,
)

from relife2 import parametric
from relife2.gammaprocess import ShapeFunctions
from relife2.types import FloatArray
from relife2.utils.integrations import shifted_laguerre


# pylint: disable=no-member


class DistributionFunctions(parametric.LifetimeFunctions, ABC):
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

    def mean(self) -> float:
        return 1 / self.rate

    def var(self) -> float:
        return 1 / self.rate**2

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

    def mean(self) -> float:
        return gamma(1 + 1 / self.shape) / self.rate

    def var(self) -> float:
        return gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2

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

    def mean(self) -> float:
        return np.exp(self.shape) * exp1(self.shape) / self.rate

    def var(self) -> float:
        return polygamma(1, 1).item() / self.rate**2

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

    def mean(self) -> float:
        return self.shape / self.rate

    def var(self) -> float:
        return self.shape / (self.rate**2)

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
        return np.log(1 + x**self.shape)

    def mean(self) -> float:
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(f"Expectancy only defined for c > 1: c = {self.shape}")
        return b / (self.rate * np.sin(b))

    def var(self) -> float:
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(f"Variance only defined for c > 2: c = {self.shape}")
        return (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))

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


class GPDistributionFunctions(parametric.LifetimeFunctions):
    """BLABLABLA"""

    def __init__(
        self,
        shape_function: ShapeFunctions,
        rate: Optional[float] = None,
        initial_resistance: Optional[float] = None,
        load_threshold: Optional[float] = None,
    ):
        super().__init__(rate=rate)
        self.add_functions("shape_function", shape_function)

        if initial_resistance is None:
            initial_resistance = np.random.uniform(1, 2, 1)

        if load_threshold is None:
            load_threshold = np.random.uniform(0, 1, 1)

        self.initial_resistance = initial_resistance
        self.load_threshold = load_threshold

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

    @property
    def resistance_magnitude(self):
        return (self.initial_resistance - self.load_threshold) * self.rate

    def init_params(self, *args: Any) -> FloatArray:
        pass

    @property
    def params_bounds(self) -> Bounds:
        return None

    def pdf(self, time: FloatArray) -> Union[float, FloatArray]:
        """BLABLABLA"""

        res = -self.shape_function.jac_nu(time) * self.moore_jac_uppergamma_c(time)

        return np.where(
            time == 0,
            int(self.shape_power == 1)
            * (
                -self.shape_rate
                * expi(-(self.initial_resistance - self.load_threshold) * self.rate)
            ),
            res,
        )

    def sf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return gammainc(
            self.shape_function.nu(time),
            (self.initial_resistance - self.load_threshold) * self.rate,
        )

    def _series_expansion(self, shape_values: FloatArray, tol: float) -> FloatArray:

        # resistance_values - shape : (n,)

        # shape : (n,)
        r = self.resistance_magnitude / (1 + shape_values)

        # shape : (n,)
        f = np.exp(
            shape_values * np.log(self.resistance_magnitude)
            - loggamma(shape_values + 1)
            - self.resistance_magnitude
        )
        # shape : (n,)
        d_f = f * (np.log(self.resistance_magnitude) - digamma(shape_values + 1))
        # shape : (n,)
        epsilon = tol / (abs(f) + abs(d_f))
        # shape : (n,)
        delta = (1 - r) * epsilon / 2

        # shape : (n,)
        n1 = np.ceil((np.log(epsilon) + np.log(1 - r)) / np.log(r), dtype=np.int32)
        n2 = np.ceil(1 + r / (1 - r), dtype=np.int32)
        n3 = np.ceil(
            np.real(lambertw(np.log(r) * delta, k=-1) / np.log(r)), dtype=np.int32
        )

        # M = max(n1, n2, 3)
        # shape : (n, M)
        range_grid = (
            np.tile(np.arange(1, np.max(n1, n2, n3) + 1), (len(n1), 1))
            + shape_values[:, None]
        )
        mask = np.ones_like(range_grid)

        # shape : (n,), fill range_grid with zeros when crossed max upper bound on conditioned indices
        ind = np.log(r) * delta >= -1 / np.exp(1)

        # shape : (n, M)
        mask[ind] = (
            mask[ind]
            <= np.maximum(np.max(np.vstack((n1, n2)), axis=0), n3)[ind][:, None]
        ) * mask[ind]
        mask[~ind] = (
            mask[~ind] <= np.max(np.vstack((n1, n2)), axis=0)[ind][:, None]
        ) * mask[~ind]

        # shape : (n, M + 1)
        harmonic = np.hstack(
            (np.zeros((range_grid.shape[0], 1)), 1 / range_grid), dtype=range_grid.dtype
        )
        cn = np.hstack(
            (np.ones((range_grid.shape[0], 1)), self.resistance_magnitude / range_grid),
            dtype=range_grid.dtype,
        )

        # shape : (n, M + 1)
        harmonic = np.cumsum(harmonic, axis=1) * mask
        cn = np.cumprod(cn, axis=1) * mask

        cn_derivative = -cn * harmonic

        # shape : (n,)
        s = np.sum(cn, axis=1)
        d_s = np.sum(cn_derivative, axis=1)

        # return shape : (n,)
        return s * d_f + f * d_s

    def _continued_fraction_expansion(
        self, shape_values: FloatArray, tol: float
    ) -> FloatArray:

        # resistance_values - shape : (n,)

        # shape : (n, 2)
        a = np.tile(
            np.array([1, 1 + self.resistance_magnitude]),
            (len(shape_values), 1),
        )
        b = np.hstack(
            (
                (np.ones_like(shape_values) * self.resistance_magnitude)[:None],
                self.resistance_magnitude
                * (2 - shape_values[:, None] + self.resistance_magnitude),
            )
        )

        # shape : (n, 2)
        d_a = np.zeros_like(a)
        d_b = np.zeros_like(b)
        d_b[:, 1] = -self.resistance_magnitude

        # shape : (n,)
        f = np.exp(
            shape_values * np.log(self.resistance_magnitude)
            - loggamma(shape_values)
            - self.resistance_magnitude
        )
        d_f = f * (np.log(self.resistance_magnitude) - digamma(shape_values))

        s = None

        res = np.ones_like(shape_values) * 2 * tol
        k = 2

        result = np.full_like(shape_values, np.nan)
        d_result = result.copy()

        while (res > tol).any():

            ak = (k - 1) * (shape_values - k)
            bk = 2 * k - shape_values + self.resistance_magnitude

            next_a = bk * a[:, 1] + ak * a[:, 0]
            next_b = bk * b[:, 1] + ak * b[:, 0]

            next_d_a = bk * d_a[:, 1] - a[:, 1] + ak * d_a[:, 0] + (k - 1) * a[:, 0]
            next_d_b = bk * d_b[:, 1] - b[:, 1] + ak * d_b[:, 0] + (k - 1) * b[:, 0]

            next_s = next_a / next_b

            if s is not None:
                res = np.abs(next_s - s) / next_s
            k += 1

            # update
            a = np.hstack((a[:, 1], next_a))
            b = np.hstack((b[:, 1], next_b))
            d_a = np.hstack((d_a[:, 1], next_d_a))
            d_b = np.hstack((d_b[:, 1], next_d_b))
            result = np.where(res <= tol, next_s, result)
            d_result = np.where(
                res <= tol,
                next_b ** (-2) * (next_b * next_d_a - next_a * next_d_b),
                d_result,
            )

        return -f * d_result - result * d_f

    def moore_jac_uppergamma_c(self, time: FloatArray, tol: float = 1e-6) -> FloatArray:
        """BLABLABLA"""

        # /!\ consider time as masked array
        shape_values = np.ravel(self.shape_function.nu(time).astype(float))
        series_indices = np.where(
            np.logical_or(
                np.logical_and(
                    shape_values <= self.resistance_magnitude,
                    self.resistance_magnitude <= 1,
                ),
                self.resistance_magnitude < shape_values,
            )
        )[0]

        result = time.copy()
        result[series_indices] = self._series_expansion(
            shape_values[series_indices], tol
        )
        result[~series_indices] = self._continued_fraction_expansion(
            shape_values[~series_indices], tol
        )
        return np.where(time == 0, 0, result)
