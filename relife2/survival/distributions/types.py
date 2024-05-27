"""
This module defines fundamental types used in distributions package

Copyright (c) 2022, RTE (https://www.rte-france.com)
See AUTHORS.txt
SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
"""

import copy
from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds

from relife2.survival.data import (
    MeasuresFactory,
    MeasuresFactoryFrom1D,
    MeasuresFactoryFrom2D,
)
from relife2.survival.parameters import Parameters

IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
FloatArray = NDArray[np.float64]


class Distribution(ABC):
    """
    Client object used to create instance of distribution model
    Object used as facade design pattern
    """

    def __init__(self):
        pass

    @abstractmethod
    def sf(self, time: ArrayLike, **kwparams: float) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def isf(
        self, probability: ArrayLike, **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (Union[float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def hf(self, time: ArrayLike, **kwparams: float) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def chf(self, time: ArrayLike, **kwparams: float) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def cdf(self, time: ArrayLike, **kwparams: float) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def pdf(
        self, probability: ArrayLike, **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (Union[float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ppf(self, time: ArrayLike, **kwparams: float) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mrl(self, time: ArrayLike, **kwparams: float) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ichf(
        self,
        cumulative_hazard_rate: ArrayLike,
        **kwparams: float,
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            cumulative_hazard_rate (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None, **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            size (Optional[int]): BLABLABLABLA
            seed (Optional[int]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mean(self, **kwparams: float) -> float:
        """
        BLABLABLABLA
        Args:
            **kwparams (float): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """

    @abstractmethod
    def var(self, **kwparams: float) -> float:
        """
        BLABLABLABLA
        Args:
            **kwparams (float): BLABLABLABLA

        Returns:
            float: BLABLABLABLA
        """

    @abstractmethod
    def fit(
        self,
        time: ArrayLike,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        **indicators: Optional[ArrayLike],
    ) -> Union[None, Parameters]:
        """
        BLABLABLABLA
        Args:
            time (Union[ArrayLike, FloatArray]):
            entry (Optional[Union[ArrayLike, FloatArray]]):
            departure (Optional[Union[ArrayLike, FloatArray]]):
            **indicators (Union[ArrayLike, FloatArray]):

        Returns:
            Union[None, Parameters]: BLABLABLABLA
        """


class DistributionFunctions(ABC):
    """
    Object that computes every probability functions of a distribution model
    """

    def __init__(self, **kparam_names: Union[float, None]):
        self.params = Parameters(**kparam_names)

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
        time: FloatArray,
        entry: Optional[FloatArray] = None,
        departure: Optional[FloatArray] = None,
        **indicators: Union[IntArray, BoolArray],
    ):

        self.functions = copy.copy(functions)
        factory: MeasuresFactory
        if time.shape[-1] == 1:
            factory = MeasuresFactoryFrom1D(time, entry, departure, **indicators)
        else:
            factory = MeasuresFactoryFrom2D(time, entry, departure, **indicators)
        (
            self.complete_lifetimes,
            self.left_censorships,
            self.right_censorships,
            self.interval_censorship,
            self.left_truncations,
            self.right_truncations,
        ) = factory()

    @abstractmethod
    def negative_log_likelihood(self) -> float:
        """
        BLABLABLABLA
        Args:
            time (): BLABLABLABLA

        Returns:
            FloatArray: BLABLABLABLA
        """

    @abstractmethod
    def jac_negative_log_likelihood(self) -> FloatArray:
        """
        BLABLABLABLA
        Args:
            time (): BLABLABLABLA

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
