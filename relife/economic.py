from abc import ABC, abstractmethod

import numpy as np

from relife.utils import reshape_1d_arg


class Reward(ABC):
    @abstractmethod
    def conditional_expectation(self, time):
        """Conditional expected reward"""
        pass

    @abstractmethod
    def sample(self, time):
        """Reward conditional sampling.

        Parameters
        ----------
        time : ndarray
            Time duration values.


        Returns
        -------
        ndarray
            Random drawing of a reward with respect to time.
        """


class RunToFailureReward(Reward):
    r"""Run-to-failure reward.

    Parameters
    ----------
    cf : float or 1darray
        The cost of failure.

    Attributes
    ----------
    cf
    """

    def __init__(self, cf):
        self.cf = reshape_1d_arg(cf)

    def conditional_expectation(self, time):
        return np.ones_like(time) * self.cf

    def sample(self, time):
        return self.conditional_expectation(time)


class AgeReplacementReward(Reward):
    r"""Age replacement reward.

    Parameters
    ----------
    cf : float or 1darray
        The cost of failure.
    cp : float or 1darray
        The cost of preventive replacement.

    Attributes
    ----------
    cf
    cp
    ar
    """

    def __init__(self, cf, cp, ar):
        self.cf = reshape_1d_arg(cf)
        self.cp = reshape_1d_arg(cp)
        self.ar = reshape_1d_arg(ar)

    def conditional_expectation(self, time):
        return np.where(time < self.ar, self.cf, self.cp)

    def sample(self, time):
        return self.conditional_expectation(time)


class Discounting(ABC):
    @abstractmethod
    def factor(self): ...

    @abstractmethod
    def annuity_factor(self, timeline): ...


class ExponentialDiscounting(Discounting):
    """
    Exponential discounting.

    Parameters
    ----------
    rate : float
        The discounting rate
    """
    def __init__(self, rate=0.0):
        self.rate = rate

    def factor(self, timeline):
        if self.rate != 0.0:
            return np.exp(-self.rate * timeline)
        else:
            return np.ones_like(timeline)

    def annuity_factor(self, timeline):
        if self.rate != 0.0:
            return (1 - np.exp(-self.rate * timeline)) / self.rate
        else:
            return timeline
