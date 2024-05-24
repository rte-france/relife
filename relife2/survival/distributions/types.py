"""
This module defines fundamental types used in distributions package
"""

from abc import ABC, abstractmethod
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from relife2.survival.parameters import Parameters

FloatArray = NDArray[np.float64]


class Distribution(ABC):
    """
    Client object used to create instance of distribution model
    Object used as facade design pattern
    """

    def __init__(self):
        pass

    @abstractmethod
    def sf(
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
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
        self, probability: Union[float | ArrayLike | FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (Union[float | ArrayLike | FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def hf(
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def chf(
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def cdf(
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
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
        self, probability: Union[float | ArrayLike | FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            probability (Union[float | ArrayLike | FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def ppf(
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
            **kwparams (float): BLABLABLABLA

        Returns:
            Union[float, FloatArray]: BLABLABLABLA
        """

    @abstractmethod
    def mrl(
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
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
        self, time: Union[int, float, ArrayLike, FloatArray], **kwparams: float
    ) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (Union[int, float, ArrayLike, FloatArray]): BLABLABLABLA
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
        time: Union[ArrayLike, FloatArray],
        entry: Optional[Union[ArrayLike, FloatArray]] = None,
        departure: Optional[Union[ArrayLike, FloatArray]] = None,
        **indicators: Union[ArrayLike, FloatArray],
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

    def __init__(self, *param_names: str, **kparam_names: float):
        self.params = Parameters(*param_names, **kparam_names)

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
    def ichf(self, time: FloatArray) -> Union[float, FloatArray]:
        """
        BLABLABLABLA
        Args:
            time (FloatArray): BLABLABLABLA

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
