"""
This module defines fundamental types used in distributions package

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union, Type

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from relife2.survival.data import (
    array_factory,
    LifetimeDataFactory,
    LifetimeDataFactoryFrom1D,
    LifetimeDataFactoryFrom2D,
    ObservedLifetimes,
    Truncations,
)
from relife2.survival.parameters import Parameters

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class DistributionFunctions(ABC):
    """
    Object that computes every probability functions of a distribution model
    """

    def __init__(self, **kparam_names: Union[float, None]):
        self.params = Parameters(**kparam_names)

    @property
    def support_upper_bound(self):
        """
        Returns:
            BLABLABLA
        """
        return np.inf

    @property
    def support_lower_bound(self):
        """
        Returns:
            BLABLABLA
        """
        return 0.0

    @abstractmethod
    def hf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def chf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def jac_hf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def jac_chf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def dhf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ichf(self, cumulative_hazard_rate: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mrl(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mean(self) -> float:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """

    @abstractmethod
    def var(self) -> float:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """

    def sf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.exp(-self.chf(time))

    def isf(self, probability: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)

    def cdf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return 1 - self.sf(time)

    def pdf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return self.hf(time) * self.sf(time)

    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        generator = np.random.RandomState(seed=seed)
        probability = generator.uniform(size=size)
        return self.isf(probability)

    def ppf(self, probability: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (FloatArray): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return self.isf(1 - probability)

    def median(self) -> float:
        """
        BLABLABLABLA
        Returns:
            float: BLABLABLABLA
        """
        return float(self.ppf(np.array(0.5)))


class DistributionLikelihood(ABC):
    """
    Object that computes every likelihood functions of a distribution model
    """

    default_hess_scheme: str = "cs"

    def __init__(
        self,
        functions: DistributionFunctions,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
    ):

        self.functions = copy.copy(functions)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

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


class DistributionOptimizer(ABC):
    """
    Object that optimize parameters of a distribution model given a likelihood
    """

    method: str = "L-BFGS-B"

    def __init__(self, likelihood: DistributionLikelihood):
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


class Distribution:
    """
    Client object used to create instance of distribution model
    Object used as facade design pattern
    """

    def __init__(
        self,
        functions: DistributionFunctions,
        likelihood: Type[DistributionLikelihood],
        optimizer: Type[DistributionOptimizer],
    ):
        self.functions = functions
        self.likelihood = likelihood
        self.optimizer = optimizer

    @property
    def params(self) -> Parameters:
        return self.functions.params

    def sf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.sf(array_factory(time)))[()]

    def isf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.isf(array_factory(probability)))[()]

    def hf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.hf(array_factory(time)))[()]

    def chf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.chf(array_factory(time)))[()]

    def cdf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.cdf(array_factory(time)))[()]

    def pdf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.pdf(array_factory(probability)))[()]

    def ppf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.ppf(array_factory(time)))[()]

    def mrl(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.mrl(array_factory(time)))[()]

    def ichf(
        self,
        cumulative_hazard_rate: ArrayLike,
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (ArrayLike): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.ichf(array_factory(cumulative_hazard_rate)))[
            ()
        ]

    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """
        return np.squeeze(self.functions.rvs(size=size, seed=seed))[()]

    def mean(self) -> float:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return self.functions.mean()

    def var(self) -> float:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return self.functions.var()

    def median(self) -> float:
        """
        BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """
        return self.functions.median()

    def fit(
        self,
        time: ArrayLike,
        inplace: bool = True,
        **kwargs: ArrayLike,
    ) -> Parameters:
        """
        BLABLABLABLA
        Args:
            time (Union[ArrayLike, FloatArray]):
            inplace ():

        Returns:
            Parameters: optimum parameters found
        """

        time = array_factory(time)
        entry = array_factory(kwargs.get("entry", np.zeros((len(time), 1))))
        departure = array_factory(
            kwargs.get("departure", np.ones((len(time), 1)) * np.inf)
        )
        lc_indicators = array_factory(
            kwargs.get("lc_indicators", np.zeros_like(time))
        ).astype(np.bool_)
        rc_indicators = array_factory(
            kwargs.get("rc_indicators", np.zeros_like(time))
        ).astype(np.bool_)

        factory: LifetimeDataFactory
        if time.shape[-1] == 1:
            factory = LifetimeDataFactoryFrom1D(
                time,
                entry=entry,
                departure=departure,
                lc_indicators=lc_indicators,
                rc_indicators=rc_indicators,
            )
        else:
            factory = LifetimeDataFactoryFrom2D(
                time,
                entry=entry,
                departure=departure,
                lc_indicators=lc_indicators,
                rc_indicators=rc_indicators,
            )
        observed_lifetimes, truncations = factory()

        likelihood_instance = self.likelihood(
            self.functions, observed_lifetimes, truncations
        )
        optimizer_instance = self.optimizer(likelihood_instance)
        optimum_params = optimizer_instance.fit()
        if inplace:
            self.functions.params.values = optimum_params.values
        return optimum_params
