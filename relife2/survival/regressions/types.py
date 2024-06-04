"""
This module defines fundamental types used in regression package

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from numpy import ma
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from relife2.survival.data import (
    LifetimeDataFactory,
    LifetimeDataFactoryFrom1D,
    LifetimeDataFactoryFrom2D,
)
from relife2.survival.distributions.types import DistributionFunctions
from relife2.survival.integrations import ls_integrate
from relife2.survival.parameters import Parameters

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class Regression(ABC):
    """
    Client object used to create instance of regression model
    Object used as facade design pattern
    """

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
            time (ArrayLike): BLABLABLABLA
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

    @abstractmethod
    def fit(
        self,
        time: ArrayLike,
        covar: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        inplace: bool = True,
        **indicators: ArrayLike,
    ) -> Parameters:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            covar (ArrayLike):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            inplace ():
            **indicators (ArrayLike):

        Returns:
            Parameters: optimum parameters found
        """


class CovarEffect(ABC):
    """
    Object that computes covariates effect functions
    """

    def __init__(self, param_values: Optional[ArrayLike] = None):
        self.params = Parameters()
        if param_values is not None:
            self.params.append(
                Parameters(
                    **{
                        f"beta_{i}": value
                        for i, value in enumerate(np.asarray(param_values))
                    }
                )
            )

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


class RegressionFunctions(ABC):
    """
    Object that computes every probability functions of a regression model
    """

    def __init__(
        self,
        baseline: DistributionFunctions,
    ):
        self.baseline = copy.deepcopy(baseline)
        self.params = Parameters()
        self.params.append(self.baseline.params)
        self.params.append(self.covar_effect.params)

    @property
    @abstractmethod
    def covar_effect(self) -> CovarEffect:
        """
        Returns:
            BLABLABLABLA
        """

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return np.inf

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLABLA
        """
        return 0.0

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
        ub = self.support_upper_bound
        mask = time >= ub
        if np.any(mask):
            time, ub = np.broadcast_arrays(time, ub)
            time = ma.MaskedArray(time, mask)
            ub = ma.MaskedArray(ub, mask)
        mu = ls_integrate(lambda x: x - time, self, time, ub, ndim=1) / self.sf(
            time, covar
        )
        return ma.filled(mu, 0)

    def moment(self, n: int) -> float:
        """
        BLABLABLA
        Args:
            n (int): BLABLABLA

        Returns:
            BLABLABLA
        """
        return ls_integrate(lambda x: x**n, self, 0, np.inf)

    def mean(self) -> float:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """
        return self.moment(1)

    def var(self) -> float:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """
        return self.moment(2) - self.moment(1) ** 2

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

    def median(self, covar: FloatArray) -> float:
        """
        BLABLABLABLA
        Args:
            covar (FloatArray): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return float(self.ppf(np.array(0.5), covar))


class RegressionLikelihood(ABC):
    """
    Object that computes every likelihood functions of a regression model
    """

    default_hess_scheme: str = "cs"

    def __init__(
        self,
        functions: RegressionFunctions,
        time: FloatArray,
        covar: FloatArray,
        entry: Optional[FloatArray] = None,
        departure: Optional[FloatArray] = None,
        **indicators: BoolArray,
    ):

        self.functions = copy.copy(functions)
        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(time, entry, departure, **indicators)
        else:
            factory = LifetimeDataFactoryFrom2D(time, entry, departure, **indicators)
        (
            self.complete_lifetimes,
            self.left_censorships,
            self.right_censorships,
            self.interval_censorship,
            self.left_truncations,
            self.right_truncations,
        ) = factory()
        self.covar = covar

    @abstractmethod
    def negative_log_likelihood(self) -> float:
        """
        BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def jac_negative_log_likelihood(self) -> FloatArray:
        """
        BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def hess_negative_log_likelihood(self) -> FloatArray:
        """
        BLABLABLABLA
        Returns:
            FloatArray: BLABLABLABLA
        """


class RegressionOptimizer(ABC):
    """
    Object that optimize parameters of a regression model given a likelihood
    """

    method: str = "L-BFGS-B"

    def __init__(self, likelihood: RegressionLikelihood):
        self.likelihood = likelihood
        self.param0 = self.init_params()
        self.bounds = self.get_params_bounds()

    @abstractmethod
    def init_params(self) -> FloatArray:
        """Init parameters values"""

    @abstractmethod
    def get_params_bounds(self) -> Bounds:
        """Returns parameters' bounds"""

    @abstractmethod
    def fit(self, **kwargs) -> Parameters:
        """Optimize model parameters to maximise likelihood"""
