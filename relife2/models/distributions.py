from typing import Any, Optional, Union

from numpy.typing import ArrayLike

from relife2.data import ObservedLifetimes, Truncations, array_factory
from relife2.functions import (
    DistributionFunctions,
    ExponentialFunctions,
    GammaFunctions,
    GompertzFunctions,
    GPDistributionFunctions,
    LikelihoodFromLifetimes,
    LogLogisticFunctions,
    PowerShapeFunctions,
    WeibullFunctions,
)
from relife2.models.core import ParametricLifetimeModel, squeeze
from relife2.typing import FloatArray


class Distribution(ParametricLifetimeModel):
    """
    Facade implementation for distribution models
    """

    functions: DistributionFunctions

    @squeeze
    def sf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """
        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.sf(time)

    @squeeze
    def isf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        return self.functions.isf(array_factory(probability))

    @squeeze
    def hf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.hf(time)

    @squeeze
    def chf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.chf(time)

    @squeeze
    def cdf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.cdf(time)

    @squeeze
    def pdf(self, probability: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            probability ():

        Returns:

        """
        probability = array_factory(probability)
        return self.functions.pdf(probability)

    @squeeze
    def ppf(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():

        Returns:

        """
        time = array_factory(time)
        return self.functions.ppf(time)

    @squeeze
    def mrl(self, time: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            time ():


        Returns:

        """
        time = array_factory(time)
        return self.functions.mrl(time)

    @squeeze
    def ichf(self, cumulative_hazard_rate: ArrayLike) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
    def rvs(
        self, size: Optional[int] = 1, seed: Optional[int] = None
    ) -> Union[float, FloatArray]:
        """

        Args:
            size ():
            seed ():

        Returns:

        """
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
    def mean(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.mean()

    @squeeze
    def var(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.var()

    @squeeze
    def median(self) -> Union[float, FloatArray]:
        """

        Returns:

        """
        return self.functions.median()

    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        return LikelihoodFromLifetimes(
            self.functions.copy(),
            observed_lifetimes,
            truncations,
        )


class Exponential(Distribution):
    """BLABLABLABLA"""

    def __init__(self, rate: Optional[float] = None):
        super().__init__(ExponentialFunctions(rate=rate))


class Weibull(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(WeibullFunctions(shape=shape, rate=rate))


class Gompertz(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GompertzFunctions(shape=shape, rate=rate))


class Gamma(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(GammaFunctions(shape=shape, rate=rate))


class LogLogistic(Distribution):
    """BLABLABLABLA"""

    functions: DistributionFunctions

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__(LogLogisticFunctions(shape=shape, rate=rate))


class GammaProcessDistribution(ParametricLifetimeModel):
    """
    BLABLABLABLA
    """

    functions: GPDistributionFunctions

    @squeeze
    def sf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.sf(time)

    @squeeze
    def isf(
        self, probability: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        probability = array_factory(probability)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.isf(probability)

    @squeeze
    def hf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.hf(time)

    @squeeze
    def chf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.chf(time)

    @squeeze
    def cdf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.cdf(time)

    @squeeze
    def pdf(
        self, probability: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            probability ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        probability = array_factory(probability)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.pdf(probability)

    @squeeze
    def ppf(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.ppf(time)

    @squeeze
    def mrl(
        self, time: ArrayLike, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            time ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        time = array_factory(time)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.mrl(time)

    @squeeze
    def ichf(
        self,
        cumulative_hazard_rate: ArrayLike,
        initial_resistance: float,
        load_threshold: float,
    ) -> Union[float, FloatArray]:
        """

        Args:
            cumulative_hazard_rate ():
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        cumulative_hazard_rate = array_factory(cumulative_hazard_rate)
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.ichf(cumulative_hazard_rate)

    @squeeze
    def rvs(
        self,
        initial_resistance: float,
        load_threshold: float,
        size: Optional[int] = 1,
        seed: Optional[int] = None,
    ) -> Union[float, FloatArray]:
        """

        Args:
            initial_resistance ():
            load_threshold ():
            size ():
            seed ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.rvs(size=size, seed=seed)

    @squeeze
    def mean(
        self, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            initial_resistance ():
            load_threshold ():

        Returns:

        """
        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.mean()

    @squeeze
    def var(
        self, initial_resistance: float, load_threshold: float
    ) -> Union[float, FloatArray]:
        """

        Args:
            initial_resistance ():
            load_threshold ():

        Returns:

        """

        self.functions.initial_resistance = float(initial_resistance)
        self.functions.load_threshold = float(load_threshold)
        return self.functions.var()

    def _init_likelihood(
        self,
        observed_lifetimes: ObservedLifetimes,
        truncations: Truncations,
        **kwargs: Any,
    ) -> LikelihoodFromLifetimes:
        if "initial_resistance" not in kwargs:
            raise ValueError(
                """
                GammaProcessDistribution likelihood expects initial_resistance as data.
                Please add initial_resistance value to kwargs.
                """
            )
        if "load_threshold" not in kwargs:
            raise ValueError(
                """
                GammaProcessDistribution likelihood expects load_threshold as data.
                Please add load_threshold value to kwargs.
                """
            )

        optimized_functions = self.functions.copy()
        optimized_functions.initial_resistance = kwargs["initial_resistance"]
        optimized_functions.load_threshold = kwargs["load_threshold"]

        return LikelihoodFromLifetimes(
            optimized_functions,
            observed_lifetimes,
            truncations,
        )


class PowerGPDistribution(GammaProcessDistribution):
    """BLABLABLABLA"""

    def __init__(
        self,
        rate: Optional[float] = None,
        shape_rate: Optional[float] = None,
        shape_power: Optional[float] = None,
    ):

        super().__init__(
            GPDistributionFunctions(
                PowerShapeFunctions(shape_rate=shape_rate, shape_power=shape_power),
                rate,
            )
        )
