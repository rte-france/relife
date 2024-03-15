from abc import ABC, abstractmethod

import numpy as np
from scipy.optimize import root_scalar

from ..core.parameter import Parameter
from ..data import DataBook


class ParametricFunction(ABC):
    def __init__(self, params: Parameter):
        if not isinstance(params, Parameter):
            raise ValueError("params must be instance of Params")
        self.params = params

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


class ParametricLikelihood(ABC):
    def __init__(self, databook: DataBook):
        if not isinstance(databook, DataBook):
            raise TypeError("ParametricLikelihood expects databook instance")
        self.databook = databook

    @abstractmethod
    def negative_log_likelihood(self) -> float:
        pass

    @abstractmethod
    def jac_negative_log_likelihood(self) -> np.ndarray:
        pass

    @abstractmethod
    def hess_negative_log_likelihood(self) -> np.ndarray:
        pass


class ParametricOptimizer(ABC):
    def __init__(self, likelihood: ParametricLikelihood):
        if not isinstance(likelihood, ParametricLikelihood):
            raise TypeError("expected ParametricLikelihood")
        self.likelihood = likelihood

    @abstractmethod
    def fit(self) -> None:
        pass
