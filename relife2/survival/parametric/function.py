from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root_scalar

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

    def isf(self, probability: np.ndarray, params: np.ndarray):
        """Approx of isf using scipy.optimize in case it is not defined in subclass functions"""
        res = root_scalar(
            lambda x: self.sf(x, params) - probability,
            method="newton",
            x0=0.0,
        )
        return res.root

    # relife/model.LifetimeModel
    def ppf(self, params: np.ndarray, probability: np.ndarray):
        return self.isf(1 - probability, params)

    # relife/model.LifetimeModel
    def median(self, params: np.ndarray):
        return self.ppf(params, 0.5)

    # relife/model.LifetimeModel
    def rvs(
        self, params: np.ndarray, size: int = 1, random_state: int = None
    ) -> np.ndarray:
        probabilities = np.random.RandomState(seed=random_state).uniform(
            size=size
        )
        return self.isf(params, probabilities)

    # relife/model.LifetimeModel
    def ls_integrate(self):
        pass

    # relife/model.LifetimeModel
    def moment(self):
        """depends upon ls_integrate"""
        pass

    # relife/model.LifetimeModel
    def mean(self):
        """depends upon ls_integrate IF NOT specified in subclass"""
        pass

    # relife/model.LifetimeModel
    def var(self):
        """depends upon ls_integrate IF NOT specified in subclass"""
        pass

    # relife/model.LifetimeModel
    def mrl(self):
        """depends upon ls_integrate IF NOT specified in subclass"""
        pass


class ParametricDistriFunction(ParametricFunction):
    def __init__(self, nb_param: int):
        self.nb_param = nb_param

    # relife/parametric.ParametricLifetimeModel
    def sf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Parametric survival function."""
        return np.exp(-self.chf(time, params))

    # relife/parametric.ParametricLifetimeModel
    def cdf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Parametric cumulative distribution function."""
        return 1 - self.sf(time, params)

    # relife/parametric.ParametricLifetimeModel
    def pdf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        """Parametric probability density function."""
        return self.hf(time, params) * self.sf(time, params)

    @abstractmethod
    def mean(self):
        """only mandatory for ParametricDistri as exact expression is known"""
        pass

    @abstractmethod
    def var(self):
        """only mandatory for ParametricDistri as exact expression is known"""
        pass

    @abstractmethod
    def mrl(self):
        """only mandatory for ParametricDistri as exact expression is known"""
        pass

    @abstractmethod
    def isf(self):
        """only mandatory for ParametricDistri as exact expression is known"""
        pass


class ExponentialDistriFunction(ParametricDistriFunction):
    def __init__(self):
        super().__init__(nb_params=1)

    # relife/distribution.Exponential
    # mandatory
    def hf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        rate = params[0]
        return rate * np.ones_like(time)

    # relife/distribution.Exponential
    # mandatory
    def chf(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        rate = params[0]
        return rate * time

    # relife/distribution.Exponential
    # mandatory
    def mean(self, params: np.ndarray) -> np.ndarray:
        rate = params[0]
        return 1 / rate

    # relife/distribution.Exponential
    # mandatory
    def var(
        self,
        params: np.ndarray,
    ) -> np.ndarray:
        rate = params[0]
        return 1 / rate**2

    # relife/distribution.Exponential
    # mandatory
    def mrl(self, time: np.ndarray, params: np.ndarray) -> np.ndarray:
        rate = params[0]
        return 1 / rate * np.ones_like(time)

    # relife/distribution.Exponential /!\ dependant of _ichf (why : carry fitted params and params)
    def ichf(
        self, cumulative_hazard_rate: np.ndarray, params: np.ndarray
    ) -> np.ndarray:
        rate = params[0]
        return cumulative_hazard_rate / rate

    # relife/model.AbsolutelyContinuousLifetimeModel /!\ dependant of ichf and _ichf
    # /!\ mathematically -np.log(probability) = cumulative_hazard_rate
    # mandatory
    def isf(
        self,
        probability: np.ndarray,
        params: np.ndarray,
    ) -> np.ndarray:

        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate, params)
