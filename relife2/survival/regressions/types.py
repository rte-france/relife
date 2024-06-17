"""
This module defines fundamental types used in regression package

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from numpy import ma
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from relife2.survival.data import ObservedLifetimes, Truncations, LifetimeData
from relife2.survival.integrations import gauss_legendre, quad_laguerre
from relife2.survival.parameters import Parameters
from relife2.survival.types import Likelihood, Functions, Model, CompositionFunctions

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class CovarEffect(Functions, ABC):
    """
    Object that computes covariates effect functions
    """

    def __init__(self, **beta: Union[float, None]):
        super().__init__(Parameters(**beta))

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return -np.inf

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return np.inf

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.params.size, self.support_lower_bound),
            np.full(self.params.size, self.support_upper_bound),
        )

    def initial_params(self) -> FloatArray:
        return np.zeros(self.params.size)

    @abstractmethod
    def g(self, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def jac_g(self, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """


class RegressionFunctions(CompositionFunctions, ABC):
    """
    Object that computes every probability functions of a regression model
    """

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return 0.0

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return np.inf

    def initial_params(self, rlc: LifetimeData) -> FloatArray:
        """initialization of params values given observed lifetimes"""

        return np.concatenate(
            (
                self.covar_effect.initial_params(),
                self.baseline.initial_params(rlc),
            )
        )

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        lb = np.concatenate(
            (
                self.covar_effect.params_bounds.lb,
                self.baseline.params_bounds.lb,
            )
        )
        ub = np.concatenate(
            (
                self.covar_effect.params_bounds.ub,
                self.baseline.params_bounds.ub,
            )
        )
        return Bounds(lb, ub)

    @abstractmethod
    def hf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def chf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def jac_hf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def jac_chf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def dhf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ichf(
        self, cumulative_hazard_rate: FloatArray, covar: FloatArray
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    def mrl(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        masked_time: ma.MaskedArray = ma.MaskedArray(
            time, time >= self.support_upper_bound
        )
        upper_bound = np.broadcast_to(
            np.asarray(self.isf(np.array(1e-4), covar)), time.shape
        )
        masked_upper_bound: ma.MaskedArray = ma.MaskedArray(
            upper_bound, time >= self.support_upper_bound
        )

        def integrand(x):
            return (x - masked_time) * self.pdf(x, covar)

        integration = gauss_legendre(
            integrand,
            masked_time,
            masked_upper_bound,
            ndim=2,
        ) + quad_laguerre(
            integrand,
            masked_upper_bound,
            ndim=2,
        )
        mrl = integration / self.sf(masked_time, covar)
        return ma.filled(mrl, 0)

    def moment(self, n: int, covar: FloatArray) -> FloatArray:
        """
        BLABLABLA
        Args:
            n (int): BLABLABLA
            covar (FloatArray):

        Returns:
            BLABLABLA
        """
        upper_bound = self.isf(np.array(1e-4), covar)

        def integrand(x):
            return x**n * self.pdf(x, covar)

        return gauss_legendre(
            integrand, np.array(0.0), upper_bound, ndim=2
        ) + quad_laguerre(integrand, upper_bound, ndim=2)

    def mean(self, covar) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray):

        Returns:
            float: BLABLABLABLA
        """
        return self.moment(1, covar)

    def var(self, covar) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray):

        Returns:
            float: BLABLABLABLA
        """
        return self.moment(2, covar) - self.moment(1, covar) ** 2

    def sf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.exp(-self.chf(time, covar))

    def isf(
        self, probability: FloatArray, covar: FloatArray
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, covar)

    def cdf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return 1 - self.sf(time, covar)

    def pdf(self, time: FloatArray, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return self.hf(time, covar) * self.sf(time, covar)

    def rvs(
        self, covar: FloatArray, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray): BLABLABLABLA
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        generator = np.random.RandomState(seed=seed)
        probability = generator.uniform(size=size)
        return self.isf(probability, covar)

    def ppf(
        self, probability: FloatArray, covar: FloatArray
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA
            covar (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return self.isf(1 - probability, covar)

    def median(self, covar: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return self.ppf(np.array(0.5), covar)


class RegressionLikelihood(Likelihood, ABC):
    """
    Object that computes every likelihood functions of a regression model
    """

    default_hess_scheme: str = "cs"

    def __init__(
        self,
        functions: RegressionFunctions,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        covar: FloatArray,
    ):
        super().__init__(functions, observed_lifetimes, truncations)
        self.covar = covar

    @abstractmethod
    def jac_negative_log_likelihood(self) -> FloatArray:
        """
        BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """


class Regression(Model, ABC):
    """
    Client object used to create instance of regression model
    Object used as facade design pattern
    """

    def __init__(self, functions: RegressionFunctions):
        super().__init__(functions)

    def _check_covar_dim(self, covar: FloatArray):
        nb_covar = covar.shape[-1]
        if nb_covar != self.functions.covar_effect.params.size:
            raise ValueError(
                f"Invalid number of covar : expected {self.functions.covar_effect.params.size}, got {nb_covar}"
            )

    @property
    def baseline(self):
        return self.functions.baseline

    @property
    def covar_effect(self):
        return self.functions.covar_effect

    @abstractmethod
    def sf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def isf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def hf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def chf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ichf(
        self,
        cumulative_hazard_rate: ArrayLike,
        covar: ArrayLike,
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def cdf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def pdf(self, probability: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ppf(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA


        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mrl(self, time: ArrayLike, covar: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA
            covar (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def rvs(
        self, covar: ArrayLike, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            covar (ArrayLike): BLABLABLABLA
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mean(self, covar: ArrayLike) -> float:
        """
        BLABLABLABLA
        Args:
            covar (ArrayLike): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """

    @abstractmethod
    def var(self, covar: ArrayLike) -> float:
        """
        BLABLABLABLA
        Args:
            covar (ArrayLike): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """

    @abstractmethod
    def median(self, covar: ArrayLike) -> float:
        """
        BLABLABLABLA
        Args:
            covar (ArrayLike): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
