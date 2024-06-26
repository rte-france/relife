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

from relife2 import parametric
from relife2.types import FloatArray
from relife2.utils.integrations import shifted_laguerre

# pylint: disable=no-member


class DistributionFunctions(parametric.LifetimeFunctions, ABC):
    """
    Object that computes every probability functions of a distribution model
    """

    def init_params(self, *args: Any) -> FloatArray:
        param0 = np.ones(self.params.size)
        param0[-1] = 1 / np.median(args[0].values)
        return param0

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.params.size, np.finfo(float).resolution),
            np.full(self.params.size, np.inf),
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
        param0 = np.empty(self.params.size, dtype=np.float64)
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


#
# class GammaProcessFunctions(ABC):
#     """BLABLABLA"""
#
#     def __init__(
#         self,
#         shape_function: ShapeFunctions,
#         rate: Optional[float] = None,
#         initial_resistance: Optional[float] = None,
#         load_threshold: Optional[float] = None,
#     ):
#         self.shape_function = copy.deepcopy(shape_function)
#         self.params = Parameters(rate=rate)
#         self.params.append(shape_function.params)
#
#         if initial_resistance is None:
#             initial_resistance = np.random.uniform(1, 2, 1)
#
#         if load_threshold is None:
#             load_threshold = np.random.uniform(0, 1, 1)
#
#         self.initial_resistance = initial_resistance
#         self.load_threshold = load_threshold
#
#     @property
#     def suppert_upper_bound(self):
#         """BLABLABLA"""
#         return np.inf
#
#     @property
#     def suppert_lower_bound(self):
#         """BLABLABLA"""
#         return 0.0
#
#     def pdf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """BLABLABLA"""
#
#         res = -self.shape_function.jac_nu(time) * self.moore_jac_uppergamma_c(time)
#
#         return np.where(
#             time == 0,
#             int(self.params.shape_power == 1)
#             * (
#                 -self.params.shape_rate
#                 * expi(
#                     -(self.initial_resistance - self.load_threshold) * self.params.rate
#                 )
#             ),
#             res,
#         )
#
#     def sf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         return gammainc(
#             self.shape_function.nu(time),
#             (self.initial_resistance - self.load_threshold) * self.params.rate,
#         )
#
#     def hf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         return self.pdf(time) / self.sf(time)
#
#     def chf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         return -np.log(self.sf(time))
#
#     @abstractmethod
#     def jac_hf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#
#     @abstractmethod
#     def jac_chf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#
#     @abstractmethod
#     def ichf(self, cumulative_hazard_rate: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             cumulative_hazard_rate (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#
#     @abstractmethod
#     def mrl(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#
#     @abstractmethod
#     def mean(self) -> float:
#         """
#         BLABLABLABLA
#         Returns:
#             float: BLABLABLABLA
#         """
#
#     @abstractmethod
#     def var(self) -> float:
#         """
#         BLABLABLABLA
#         Returns:
#             float: BLABLABLABLA
#         """
#
#     def moore_jac_uppergamma_c(self, time, tol=1e-6, print_feedback=False):
#         """BLABLABLA"""
#
#         ind0 = np.where(time == 0)[0]
#         non0_times = np.delete(time, ind0)
#         x = (self.initial_resistance - self.load_threshold) * self.params.rate
#
#         # critère d'arrêt calculé pour une intégrale voisine, et non cible.
#         # Ne devrait pas changer grand chose mais à modifier.
#
#         p_ravel = np.ravel(self.shape_function.nu(non0_times)).astype(float)
#         logic_one = np.logical_and(p_ravel <= x, x <= 1)
#         logic_two = x < p_ravel
#
#         series_indices = np.where(np.logical_or(logic_one, logic_two))[0]
#
#         # if((p<=x<=1) | (x<p)): # On this case we use the series expansion of the incomplete gamma
#         result = []
#         for i in range(len(p_ravel)):
#             p = p_ravel[i]
#             if i in series_indices:
#
#                 # Initialization of parameters
#                 r = x / (1 + p)
#
#                 # f = np.exp(-x) * (x ** p) / sc.gamma(p + 1)
#                 # d_f = np.exp(-x) * x ** p * (np.log(x) - digamma(p + 1)) / sc.gamma(p + 1)
#                 f = np.exp(p * np.log(x) - loggamma(p + 1) - x)
#                 d_f = f * (np.log(x) - digamma(p + 1))
#                 epsilon = tol / (abs(f) + abs(d_f))
#                 delta = (1 - r) * epsilon / 2
#
#                 # determining stopping criteria for the infinite series s and dS:
#                 n1 = np.ceil((np.log(epsilon) + np.log(1 - r)) / np.log(r)).astype(int)
#
#                 n2 = np.ceil(1 + r / (1 - r)).astype(int)
#
#                 if np.log(r) * delta >= -1 / np.exp(1):
#                     n3 = np.ceil(
#                         np.real(lambertw(np.log(r) * delta, k=-1) / np.log(r))
#                     ).astype(int)
#                     n = max(n1, n2, n3)
#                 else:
#                     n = max(n1, n2)
#
#                 # Computing the coefficients C_n and their derivatives
#                 cn = x / (p + np.arange(1, n + 1))
#                 cn = np.insert(cn, 0, 1)
#                 cn = cn.cumprod()
#
#                 harmonic = 1 / (p + np.arange(1, n + 1))
#                 harmonic = np.insert(harmonic, 0, 0)
#                 harmonic = np.cumsum(harmonic)
#
#                 cn_derivative = -cn * harmonic
#
#                 s = sum(cn)
#                 d_s = sum(cn_derivative)
#
#                 if print_feedback:
#                     print(
#                         f"Series expansion was used. Convergence happened after {n} steps"
#                     )
#
#                 # result.append(sc.gamma(p) * digamma(p) - s * d_f - f * d_s)
#                 result.append(s * d_f + f * d_s)
#
#             else:  # On this case we use the continued fraction expansion of the incomplete gamma
#
#                 # Parameter initialization
#                 a = [1, 1 + x]
#                 b = [x, x * (2 - p + x)]
#                 d_a = [0, 0]
#                 d_b = [0, -x]
#
#                 # f = np.exp(-x) * x ** p / sc.gamma(p)
#                 # d_f = np.exp(-x) * x ** p * (np.log(x) - digamma(p)) / sc.gamma(p)
#                 f = np.exp(p * np.log(x) - loggamma(p) - x)
#                 d_f = f * (np.log(x) - digamma(p))
#
#                 s = []
#                 res = 2 * tol
#                 k = 2
#                 while res > tol:
#
#                     ak = (k - 1) * (p - k)
#                     bk = 2 * k - p + x
#
#                     a.append(bk * a[k - 1] + ak * a[k - 2])
#                     b.append(bk * b[k - 1] + ak * b[k - 2])
#
#                     d_a.append(
#                         bk * d_a[k - 1]
#                         - a[k - 1]
#                         + ak * d_a[k - 2]
#                         + (k - 1) * a[k - 2]
#                     )
#                     d_b.append(
#                         bk * d_b[k - 1]
#                         - b[k - 1]
#                         + ak * d_b[k - 2]
#                         + (k - 1) * b[k - 2]
#                     )
#
#                     s.append(a[-1] / b[-1])
#
#                     if len(s) > 1:
#                         res = abs(s[-1] - s[-2]) / s[-1]
#                     k += 1
#
#                 s = s[-1]
#                 d_s = b[-1] ** (-2) * (b[-1] * d_a[-1] - a[-1] * d_b[-1])
#
#                 if print_feedback:
#                     print(
#                         f"Continued fraction expansion was used. Convergence happened after {k - 1} steps"
#                     )
#
#                 result.append(-f * d_s - s * d_f)
#
#         result = np.array(result).reshape(self.shape_function.nu(non0_times).shape)
#         return np.insert(result, ind0, 0)
#
#     def isf(self, probability: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             probability (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         cumulative_hazard_rate = -np.log(probability)
#         return self.ichf(cumulative_hazard_rate)
#
#     def cdf(self, time: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             time (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         return 1 - self.sf(time)
#
#     def rvs(
#         self, size: int = 1, seed: Optional[int] = None
#     ) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             size (Optional[int]): BLABLABLABLA
#             seed (Optional[int]): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         generator = np.random.RandomState(seed=seed)
#         probability = generator.uniform(size=size)
#         return self.isf(probability)
#
#     def ppf(self, probability: FloatArray) -> Union[float, FloatArray]:
#         """
#         BLABLABLABLA
#         Args:
#             probability (FloatArray): BLABLABLABLA
#
#         Returns:
#             Union[float, FloatArray]: BLABLABLABLA
#         """
#         return self.isf(1 - probability)
#
#     def median(self) -> float:
#         """
#         BLABLABLABLA
#         Returns:
#             float: BLABLABLABLA
#         """
#         return float(self.ppf(np.array(0.5)))
