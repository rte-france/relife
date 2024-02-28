from abc import ABC, abstractmethod
from typing import Type

import numpy as np
from scipy.optimize import root_scalar

from ..parameter import Parameter


class ParametricFunction(ABC):
    def __init__(self, params: Type[Parameter]):
        if not isinstance(params, Parameter):
            raise ValueError("params must be instance of Params")
        self.params = params

    def _sanity_check(self):
        """run necessary functions with random array to check
        if they all run (especially if params args pass)
        """
        pass

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

    def isf(self, probability: np.ndarray):
        """Approx of isf using scipy.optimize in case it is not defined in subclass functions"""
        res = root_scalar(
            lambda x: self.sf(x) - probability,
            method="newton",
            x0=0.0,
        )
        return res.root

    # relife/model.LifetimeModel
    def ppf(self, probability: np.ndarray):
        return self.isf(1 - probability)

    # relife/model.LifetimeModel
    def median(self):
        return self.ppf(0.5)

    # relife/model.LifetimeModel
    def rvs(self, size: int = 1, random_state: int = None) -> np.ndarray:
        probabilities = np.random.RandomState(seed=random_state).uniform(
            size=size
        )
        return self.isf(probabilities)

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
    def __init__(self, nb_params: int = None, param_names: list = None):
        params = Parameter(nb_params=nb_params, param_names=param_names)
        super().__init__(params)

    # relife/parametric.ParametricLifetimeModel
    def sf(self, time: np.ndarray) -> np.ndarray:
        """Parametric survival function."""
        return np.exp(-self.chf(time))

    # relife/parametric.ParametricLifetimeModel
    def cdf(self, time: np.ndarray) -> np.ndarray:
        """Parametric cumulative distribution function."""
        return 1 - self.sf(time)

    # relife/parametric.ParametricLifetimeModel
    def pdf(self, time: np.ndarray) -> np.ndarray:
        """Parametric probability density function."""
        return self.hf(time) * self.sf(time)

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
    def __init__(self, param_names=["rate"]):
        super().__init__(param_names=param_names)

    # relife/distribution.Exponential
    # mandatory
    def hf(self, time: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return self.params.rate * np.ones_like(time)

    # relife/distribution.Exponential
    # mandatory
    def chf(self, time: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return self.params.rate * time

    # relife/distribution.Exponential
    # mandatory
    def mean(self) -> np.ndarray:
        # rate = self.params[0]
        return 1 / self.params.rate

    # relife/distribution.Exponential
    # mandatory
    def var(self) -> np.ndarray:
        # rate = self.params[0]
        return 1 / self.params.rate**2

    # relife/distribution.Exponential
    # mandatory
    def mrl(self, time: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return 1 / self.params.rate * np.ones_like(time)

    # relife/distribution.Exponential /!\ dependant of _ichf (why : carry fitted params and params)
    def ichf(self, cumulative_hazard_rate: np.ndarray) -> np.ndarray:
        # rate = self.params[0]
        return cumulative_hazard_rate / self.params.rate

    # relife/model.AbsolutelyContinuousLifetimeModel /!\ dependant of ichf and _ichf
    # /!\ mathematically -np.log(probability) = cumulative_hazard_rate
    # mandatory
    def isf(self, probability: np.ndarray) -> np.ndarray:
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)
