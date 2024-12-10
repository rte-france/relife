from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds
from scipy.special import digamma, exp1, gamma, gammaincc, gammainccinv
from typing_extensions import override

from relife2.fiability import ParametricLifetimeModel
from relife2.utils.data import LifetimeData
from relife2.utils.integration import shifted_laguerre


# Ts type var is a zero long tuple (see https://github.com/python/mypy/issues/16199)
# note : Tuple[()] behaves differently (to follow)
# no args are required
class Distribution(ParametricLifetimeModel[()], ABC):
    """
    Base class of distribution model.
    """

    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Survival function.

        The survival function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Survival function values at each given time.
        """
        return super().sf(time)

    @override
    def isf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Inverse survival function.

        The survival function of the distribution

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Complement quantile corresponding to probability.
        """
        cumulative_hazard_rate = -np.log(probability)
        return self.ichf(cumulative_hazard_rate)

    @override
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative distribution function.

        The survival function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative distribution function values at each given time.
        """
        return super().cdf(time)

    def pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Probability density function.

        The probability density function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            The probability density function evaluated at each given time.
        """
        return super().pdf(time)

    @override
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Percent point function.

        The percent point function of the distribution. It corresponds to the inverse of
        the cumulative distribution function

        Parameters
        ----------
        probability : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Quantile corresponding to probability.
        """
        return super().ppf(probability)

    @override
    def rvs(self, *, size: int = 1, seed: Optional[int] = None):
        """
        Random variable sampling.

        Parameters
        ----------
        size : int, default 1
            Sized of the generated sample.
        seed : int, default None
            Random seed.

        Returns
        -------
        ndarray of shape (size, )
            Sample of random lifetimes.
        """

        return super().rvs(size=size, seed=seed)

    @override
    def moment(self, n: int) -> NDArray[np.float64]:
        """
        n-th order moment of the distribution.

        Parameters
        ----------
        n : int
            Order of the moment, at least 1.

        Returns
        -------
        ndarray of shape (0, )
            n-th order moment of the distribution.
        """

        return super().moment(n)

    @override
    def median(self) -> NDArray[np.float64]:
        """Median of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Median value.
        """
        return super().median()

    def init_params(self, lifetime_data: LifetimeData) -> None:
        param0 = np.ones(self.nb_params)
        param0[-1] = 1 / np.median(lifetime_data.rlc.values)
        self.params = param0

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.nb_params, np.finfo(float).resolution),
            np.full(self.nb_params, np.inf),
        )

    @abstractmethod
    def jac_hf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def jac_chf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    @abstractmethod
    def dhf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]: ...

    def jac_sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the parametric survival function with respect to params."""
        return -self.jac_chf(time) * self.sf(time)

    def jac_cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the parametric cumulative distribution function with
        respect to params."""
        return -self.jac_sf(time)

    def jac_pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the parametric probability density function with respect to
        params."""
        return self.jac_hf(time) * self.sf(time) + self.jac_sf(time) * self.hf(time)


class Exponential(Distribution):
    """
    Exponential lifetime distribution.

    Parameters
    ----------
    rate : float, optional
        rate parameter


    See Also
    --------
    regression.AFT : AFT regression
    regression.ProportionalHazard : AFT regression

    Examples
    --------
    Constructing exponential model with rate of 0.75.

    >>> model = Exponential(0.75)
    >>> time = np.linspace(3, 10, num=5)
    >>> model.sf(time)
    array([0.10539922, 0.02836782, 0.00763509, 0.00205496, 0.00055308]))

    Notice that only one asset is considered here. To pass another asset, use a 2d time array

    >>> time = np.linspace([3, 5], [10, 10], num=5, axis=1)
    >>> model.sf(time)
    array([[0.10539922, 0.02836782, 0.00763509, 0.00205496, 0.00055308],
           [0.02351775, 0.00920968, 0.00360656, 0.00141235, 0.00055308]])
    """

    def __init__(self, rate: Optional[float] = None):
        """
        Parameters
        ----------
        rate : float, optional
            rate parameter
        """
        super().__init__()
        self.new_params(rate=rate)

    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.
        """
        return self.rate * np.ones_like(time)

    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.

        Notes
        -----
        The cumulative hazard function is the integral of the hazard function.
        """
        return self.rate * time

    @override
    def mean(self) -> NDArray[np.float64]:
        """Mean of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Mean value.

        Notes
        -----
        The mean of a distribution is the moment of the first order.
        """
        return np.array(1 / self.rate)

    @override
    def var(self) -> NDArray[np.float64]:
        """Variance of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Variance value.
        """
        return np.array(1 / self.rate**2)

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values.
        """
        return 1 / self.rate * np.ones_like(time)

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        Cumulative hazard rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e. time.
        """
        return cumulative_hazard_rate / self.rate

    def jac_hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        return np.ones((time.size, 1))

    def jac_chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the cumulative hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        return np.ones((time.size, 1)) * time

    def dhf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """
        return np.zeros_like(time)


class Weibull(Distribution):
    """
    Weibull lifetime distribution.

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter


    See Also
    --------
    regression.AFT : AFT regression
    regression.ProportionalHazard : AFT regression
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.
        """
        return self.shape * self.rate * (self.rate * time) ** (self.shape - 1)

    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.

        Notes
        -----
        The cumulative hazard function is the integral of the hazard function.
        """
        return (self.rate * time) ** self.shape

    @override
    def mean(self) -> NDArray[np.float64]:
        """Mean of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Mean value.

        Notes
        -----
        The mean of a distribution is the moment of the first order.
        """
        return np.array(gamma(1 + 1 / self.shape) / self.rate)

    @override
    def var(self) -> NDArray[np.float64]:
        """Variance of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Variance value.
        """
        return np.array(gamma(1 + 2 / self.shape) / self.rate**2 - self.mean() ** 2)

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values.
        """
        return (
            gamma(1 / self.shape)
            / (self.rate * self.shape * self.sf(time))
            * gammaincc(
                1 / self.shape,
                (self.rate * time) ** self.shape,
            )
        )

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        Cumulative hazard rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e. time.
        """
        return cumulative_hazard_rate ** (1 / self.shape) / self.rate

    def jac_hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        return np.column_stack(
            (
                self.rate
                * (self.rate * time) ** (self.shape - 1)
                * (1 + self.shape * np.log(self.rate * time)),
                self.shape**2 * (self.rate * time) ** (self.shape - 1),
            )
        )

    def jac_chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the cumulative hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        return np.column_stack(
            (
                np.log(self.rate * time) * (self.rate * time) ** self.shape,
                self.shape * time * (self.rate * time) ** (self.shape - 1),
            )
        )

    def dhf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """
        return (
            self.shape
            * (self.shape - 1)
            * self.rate**2
            * (self.rate * time) ** (self.shape - 2)
        )


class Gompertz(Distribution):
    """
    Gompertz lifetime distribution.

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter


    See Also
    --------
    regression.AFT : AFT regression
    regression.ProportionalHazard : AFT regression
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def init_params(self, lifetime_data: LifetimeData) -> None:
        param0 = np.empty(self.nb_params, dtype=float)
        rate = np.pi / (np.sqrt(6) * np.std(lifetime_data.rlc.values))
        shape = np.exp(-rate * np.mean(lifetime_data.rlc.values))
        param0[0] = shape
        param0[1] = rate
        self.params = param0

    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.
        """
        return self.shape * self.rate * np.exp(self.rate * time)

    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.

        Notes
        -----
        The cumulative hazard function is the integral of the hazard function.
        """
        return self.shape * np.expm1(self.rate * time)

    @override
    def mean(self) -> NDArray[np.float64]:
        """Mean of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Mean value.

        Notes
        -----
        The mean of a distribution is the moment of the first order.
        """
        return np.array(np.exp(self.shape) * exp1(self.shape) / self.rate)

    # @override
    # def var(self) -> NDArray[np.float64]:
    #     """Variance of the distribution.
    #
    #     Returns
    #     -------
    #     ndarray of shape (0,)
    #         Variance value.
    #     """
    #     return np.array(polygamma(1, 1).item() / self.rate**2)

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values.
        """
        z = self.shape * np.exp(self.rate * time)
        return np.exp(z) * exp1(z) / self.rate

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        Cumulative hazard rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e. time.
        """
        return 1 / self.rate * np.log1p(cumulative_hazard_rate / self.shape)

    def jac_hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        return np.column_stack(
            (
                self.rate * np.exp(self.rate * time),
                self.shape * np.exp(self.rate * time) * (1 + self.rate * time),
            )
        )

    def jac_chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the cumulative hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        return np.column_stack(
            (
                np.expm1(self.rate * time),
                self.shape * time * np.exp(self.rate * time),
            )
        )

    def dhf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """
        return self.shape * self.rate**2 * np.exp(self.rate * time)


class Gamma(Distribution):
    """
    Gamma lifetime distribution.

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter


    See Also
    --------
    regression.AFT : AFT regression
    regression.ProportionalHazard : AFT regression
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def _uppergamma(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return gammaincc(self.shape, x) * gamma(self.shape)

    def _jac_uppergamma_shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return shifted_laguerre(
            lambda s: np.log(s) * s ** (self.shape - 1),
            x,
            ndim=np.ndim(x),
        )

    @override
    @property
    def _default_hess_scheme(self) -> str:
        return "2-point"

    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.
        """
        x = self.rate * time
        return self.rate * x ** (self.shape - 1) * np.exp(-x) / self._uppergamma(x)

    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.

        Notes
        -----
        The cumulative hazard function is the integral of the hazard function.
        """
        x = self.rate * time
        return np.log(gamma(self.shape)) - np.log(self._uppergamma(x))

    @override
    def mean(self) -> NDArray[np.float64]:
        """Mean of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Mean value.

        Notes
        -----
        The mean of a distribution is the moment of the first order.
        """
        return np.array(self.shape / self.rate)

    @override
    def var(self) -> NDArray[np.float64]:
        """Variance of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Variance value.
        """
        return np.array(self.shape / (self.rate**2))

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        Cumulative hazard rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e. time.
        """
        return 1 / self.rate * gammainccinv(self.shape, np.exp(-cumulative_hazard_rate))

    def jac_hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        x = self.rate * time
        return (
            x ** (self.shape - 1)
            * np.exp(-x)
            / self._uppergamma(x) ** 2
            * np.column_stack(
                (
                    self.rate * np.log(x) * self._uppergamma(x)
                    - self.rate * self._jac_uppergamma_shape(x),
                    (self.shape - x) * self._uppergamma(x) + x**self.shape * np.exp(-x),
                )
            )
        )

    def jac_chf(
        self,
        time: float | NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Jacobian of the cumulative hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        x = self.rate * time
        return np.column_stack(
            (
                digamma(self.shape)
                - self._jac_uppergamma_shape(x) / self._uppergamma(x),
                x ** (self.shape - 1) * time * np.exp(-x) / self._uppergamma(x),
            )
        )

    def dhf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """
        return self.hf(time) * ((self.shape - 1) / time - self.rate + self.hf(time))

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values.
        """
        return super().mrl(time)


class LogLogistic(Distribution):
    """
    Log-logistic probability distribution.

    Parameters
    ----------
    shape : float, optional
        shape parameter
    rate : float, optional
        rate parameter


    See Also
    --------
    regression.AFT : AFT regression
    regression.ProportionalHazard : AFT regression
    """

    def __init__(self, shape: Optional[float] = None, rate: Optional[float] = None):
        super().__init__()
        self.new_params(shape=shape, rate=rate)

    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Hazard function.

        The hazard function of the distribution

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Hazard values at each given time.
        """
        x = self.rate * time
        return self.shape * self.rate * x ** (self.shape - 1) / (1 + x**self.shape)

    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Cumulative hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Cumulative hazard values at each given time.

        Notes
        -----
        The cumulative hazard function is the integral of the hazard function.
        """
        x = self.rate * time
        return np.array(np.log(1 + x**self.shape))

    @override
    def mean(self) -> NDArray[np.float64]:
        """Mean of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Mean value.

        Notes
        -----
        The mean of a distribution is the moment of the first order.
        """
        b = np.pi / self.shape
        if self.shape <= 1:
            raise ValueError(
                f"Expectancy only defined for shape > 1: shape = {self.shape}"
            )
        return np.array(b / (self.rate * np.sin(b)))

    @override
    def var(self) -> NDArray[np.float64]:
        """Variance of the distribution.

        Returns
        -------
        ndarray of shape (0,)
            Variance value.
        """
        b = np.pi / self.shape
        if self.shape <= 2:
            raise ValueError(
                f"Variance only defined for shape > 2: shape = {self.shape}"
            )
        return np.array(
            (1 / self.rate**2) * (2 * b / np.sin(2 * b) - b**2 / (np.sin(b) ** 2))
        )

    @override
    def ichf(
        self, cumulative_hazard_rate: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Inverse cumulative hazard function.

        Parameters
        ----------
        Cumulative hazard rate : float or ndarray, shape (n, ) or (m, n)
            Cumulative hazard rate values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Inverse cumulative hazard values, i.e. time.
        """
        return ((np.exp(cumulative_hazard_rate) - 1) ** (1 / self.shape)) / self.rate

    def jac_hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        x = self.rate * time
        return np.column_stack(
            (
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (1 + x**self.shape + self.shape * np.log(self.rate * time)),
                (self.rate * x ** (self.shape - 1) / (1 + x**self.shape) ** 2)
                * (self.shape**2 / self.rate),
            )
        )

    def jac_chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Jacobian of the cumulative hazard function.

        Parameters
        ----------
        Elapsed time : float or ndarray, shape (n, ) or (m, n)
            Probability values. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Jacobian values, here gradient values at each given time.
        """
        x = self.rate * time
        return np.column_stack(
            (
                (x**self.shape / (1 + x**self.shape)) * np.log(self.rate * time),
                (x**self.shape / (1 + x**self.shape)) * (self.shape / self.rate),
            )
        )

    def dhf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Derivative of the hazard function.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Derivative values with respect to time.
        """
        x = self.rate * time
        return (
            self.shape
            * self.rate**2
            * x ** (self.shape - 2)
            * (self.shape - 1 - x**self.shape)
            / (1 + x**self.shape) ** 2
        )

    @override
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        """Mean residual life.

        Parameters
        ----------
        time : float or ndarray, shape (n, ) or (m, n)
            Elapsed time. If the given shape is (n, ), one asset and n points of measure are considered
            To consider m assets with respectively n points of measure,
            pass an array of shape (m, n).

        Returns
        -------
        ndarray of shape (), (n, ) or (m, n)
            Mean residual life values.
        """
        return super().mrl(time)
