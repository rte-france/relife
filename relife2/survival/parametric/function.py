from abc import ABC, abstractmethod

import numpy as np

# from .. import SurvivalData

# class DistributionFunctions(ParametricFunction):
#     def __init__(self, nb_params):
#         super().__init__(nb_params)

#     def sf(self, dataset: SurvivalData):
#         pass

#     def hf(self, dataset: SurvivalData):
#         pass


class ParametricFunction(ABC):
    @abstractmethod
    def sf(self):
        pass

    @abstractmethod
    def cdf(self):
        pass

    @abstractmethod
    def pdf(self):
        pass

    @abstractmethod
    def hf(self):
        pass

    @abstractmethod
    def chf(self):
        pass


class ParametricDistriFunction(ParametricFunction):
    def __init__(self, nb_param: int):
        self.nb_param = nb_param

    # relife/parametric.ParametricLifetimeModel
    def sf(self, params: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        """Parametric survival function."""
        return np.exp(-self.chf(params, elapsed_time))

    # relife/parametric.ParametricLifetimeModel
    def cdf(self, params: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        """Parametric cumulative distribution function."""
        return 1 - self.sf(params, elapsed_time)

    # relife/parametric.ParametricLifetimeModel
    def pdf(self, params: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        """Parametric probability density function."""
        return self.hf(params, elapsed_time) * self.sf(params, elapsed_time)

    @abstractmethod
    def mean(self):
        pass

    @abstractmethod
    def var(self):
        pass

    @abstractmethod
    def mrl(self):
        pass


class ExponentialDistriFunction(ParametricDistriFunction):
    def __init__(self):
        super().__init__(nb_params=1)

    # relife/distribution.Exponential
    def hf(self, params: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        rate = params[0]
        return rate * np.ones_like(elapsed_time)

    # relife/distribution.Exponential
    def chf(self, params: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        rate = params[0]
        return rate * elapsed_time

    # relife/distribution.Exponential
    def mean(self, params: np.ndarray) -> np.ndarray:
        rate = params[0]
        return 1 / rate

    # relife/distribution.Exponential
    def var(
        self,
        params: np.ndarray,
    ) -> np.ndarray:
        rate = params[0]
        return 1 / rate**2

    # relife/distribution.Exponential
    def mrl(self, params: np.ndarray, elapsed_time: np.ndarray) -> np.ndarray:
        rate = params[0]
        return 1 / rate * np.ones_like(elapsed_time)

    # relife/distribution.Exponential /!\ dependant of _ichf (why : carry fitted params and params)
    def ichf(
        self, params: np.ndarray, cumulative_hazard_rate: np.ndarray
    ) -> np.ndarray:
        rate = params[0]
        return cumulative_hazard_rate / rate

    # relife/model.AbsolutelyContinuousLifetimeModel /!\ dependant of ichf and _ichf
    # /!\ mathematically -np.log(probability) = cumulative_hazard_rate
    def isf(
        self,
        params: np.ndarray,
        probability: np.ndarray,
    ) -> np.ndarray:

        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(params, cumulative_hazard_rate)
