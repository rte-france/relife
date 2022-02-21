"""Generic renewal processes."""

# Copyright (c) 2022, RTE (https://www.rte-france.com)
# See AUTHORS.txt
# SPDX-License-Identifier: Apache-2.0 (see LICENSE.txt)
# This file is part of ReLife, an open source Python library for asset
# management based on reliability theory and lifetime data analysis.

import numpy as np
from typing import Tuple

from .model import LifetimeModel, AbsolutelyContinuousLifetimeModel
from .reward import Reward
from .discounting import Discount, ExponentialDiscounting
from .data import RenewalData, RenewalRewardData
from .utils import args_size, args_take, args_ndim


def _check_absolutely_continuous(model: LifetimeModel) -> None:
    if not isinstance(model, AbsolutelyContinuousLifetimeModel):
        raise NotImplementedError("The lifetime model must be absolutely continuous")


def _check_exponential_discounting(discount: Discount) -> None:
    if not isinstance(discount, ExponentialDiscounting):
        raise NotImplementedError("The discount function must be exponential")


def renewal_equation_solver(
    F: np.ndarray, Fm: np.ndarray, y: np.ndarray, D: np.ndarray = None
) -> np.ndarray:
    r"""Renewal equation solver.

    Parameters
    ----------
    F : ndarray
        The cumulative distribution function evaluated at each point of the
        timeline.
    Fm : ndarray
        The cumulative distribution function evaluated at each midpoint of the
        timeline.
    y : ndarray
        A given function evaluated at each point of the timeline.
    D : ndarray, optional
        Discount function value at each point of the timeline, by default None.

    Returns
    -------
    ndarray
        Renewal function evaluated at each point of the timeline.

    Notes
    -----
    The timeline must start at 0.

    Solve the renewal equation for :math:`z`, such that for :math:`t\geq 0`.

    .. math::

        \begin{aligned} z(t) & = y(t) + \int_0^t z(t-x) D(x) \mathrm{d}F(x) \\
            & = y(t) + z \ast F(t)
        \end{aligned}

    where :math:`F` is the cumulative distribution function, :math:`y` a
    function and :math:`D` the exponential discounting factor.

    References
    ----------
    .. [1] Dragomir, S. S. (2011). Approximating the Riemann–Stieltjes integral
        by a trapezoidal quadrature rule with applications. Mathematical and
        computer modelling, 54(1-2), 243-260.

    .. [2] Tortorella, M. (2005). Numerical solutions of renewal-type integral
        equations. INFORMS Journal on Computing, 17(1), 66-74.
    """
    if D is None:
        D = np.ones_like(F)
    z = np.empty(y.shape)
    u = D * np.insert(F[..., 1:] - Fm, 0, 1, axis=-1)
    v = D[..., :-1] * np.insert(np.diff(Fm), 0, 1, axis=-1)
    q0 = 1 / (1 - D[..., 0] * Fm[..., 0])
    z[..., 0] = y[..., 0]
    z[..., 1] = q0 * (y[..., 1] + z[..., 0] * u[..., 1])
    for n in range(2, F.shape[-1]):
        z[..., n] = q0 * (
            y[..., n]
            + z[..., 0] * u[..., n]
            + np.sum(z[..., 1:n][..., ::-1] * v[..., 1:n], axis=-1)
        )
    return z


def delayed_renewal(
    z: np.ndarray, F1: np.ndarray, F1m: np.ndarray, y1: np.ndarray, D: np.ndarray = None
) -> np.ndarray:
    r"""Add delay for the first renewal to a solution of a renewal equation.

    Parameters
    ----------
    z : ndarray
        The solution of a renewal equation at each point of the timeline.
    F1 : ndarray
        The cumulative distribution function of the first renewal evaluated at
        each point of the timeline.
    F1m : ndarray
        The cumulative distribution function of the first renewal evaluated at
        each midpoint of the timeline.
    y : ndarray
        A given function related to the first renewal evaluated at each point of
        the timeline.
    D : ndarray, optional
        Discount function value at each point of the timeline, by default None.

    Returns
    -------
    ndarray
        Delayed solution of the renewal equation at each point of the timeline.

    Notes
    -----
    The solution of the renewal equation is delayed by computing:

    .. math::

        z_1(t) = y_1(t) + \int_0^t z(t-x) D(x) \mathrm{d}F_1(x)

    where :math:`z` is a solution of a renewal equation, :math:`D` the
    exponential discounting factor, :math:`F_1` is the cumulative distribution
    function of the first renwal and :math:`y_1` a function related to the first
    renewal.
    """
    if D is None:
        D = np.ones_like(F1)
    z1 = np.empty(y1.shape)
    u1 = D * np.insert(F1[..., 1:] - F1m, 0, 1, axis=-1)
    v1 = D[..., :-1] * np.insert(np.diff(F1m), 0, 1, axis=-1)
    z1[..., 0] = y1[..., 0]
    z1[..., 1] = (
        y1[..., 1] + z[..., 0] * u1[..., 1] + z[..., 1] * D[..., 0] * F1m[..., 0]
    )
    for n in range(2, F1.shape[-1]):
        z1[..., n] = (
            y1[..., n]
            + z[..., 0] * u1[..., n]
            + z[..., n] * D[..., 0] * F1m[..., 0]
            + np.sum(z[..., 1:n][..., ::-1] * v1[..., 1:n], axis=-1)
        )
    return z1


class RenewalProcess:
    """Renewal process."""

    def __init__(self, model: LifetimeModel, model1: LifetimeModel = None) -> None:
        """Creates a renewal process.

        Parameters
        ----------
        model : LifetimeModel
            A lifetime model representing the durations between events.

        model1 : LifetimeModel, optional
            A lifetime model for the first renewal (delayed renewal process), by
            default None.

        """
        self.model = model
        self.model1 = model1

    def renewal_function(
        self,
        t: np.ndarray,
        model_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        r"""The renewal function.

        Parameters
        ----------
        t : 1D array
            Timeline.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().

        Returns
        -------
        ndarray
            The renewal function evaluated at each point of the timeline.

        Notes
        -----

        The expected total number of renewals is computed  by solving the
        renewal equation:

        .. math::

            m(t) = F_1(t) + \int_0^t m(t-x) \mathrm{d}F(x)

        where:

        - :math:`m` is the renewal function,
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`F_1` is the cumulative distribution function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """
        tm = 0.5 * (t[1:] + t[:-1])
        F = self.model.cdf(t, *model_args)
        Fm = self.model.cdf(tm, *model_args)
        if self.model1 is not None:
            y = self.model1.cdf(t, *model1_args)
        else:
            y = F
        return renewal_equation_solver(F, Fm, y)

    def renewal_density(
        self,
        t: np.ndarray,
        model_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        r"""The renewal density.

        Parameters
        ----------
        t : 1D array
            Timeline.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().

        Returns
        -------
        ndarray
            Renewal density evaluated at each point of the timeline.

        Raises
        ------
        NotImplementedError
            If the lifetime model is not absolutely continuous.

        Notes
        -----
        The renewal density is the derivative of the renewal function with
        respect to time. It is computed by solving the renewal equation:

        .. math::

            \mu(t) = f_1(t) + \int_0^t \mu(t-x) \mathrm{d}F(x)

        where:

        - :math:`\mu` is the renewal function,
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`f_1` is the probability density function of the underlying
          lifetime model for the fist renewal in the case of a delayed renewal
          process.

        References
        ----------
        .. [1] Rausand, M., Barros, A., & Hoyland, A. (2020). System Reliability
            Theory: Models, Statistical Methods, and Applications. John Wiley &
            Sons.
        """
        if self.model1 is not None:
            _check_absolutely_continuous(self.model1)
            y = self.model1.pdf(t, *model1_args)
        else:
            _check_absolutely_continuous(self.model)
            y = self.model.pdf(t, *model_args)
        tm = 0.5 * (t[1:] + t[:-1])
        F = self.model.cdf(t, *model_args)
        Fm = self.model.cdf(tm, *model_args)
        return renewal_equation_solver(F, Fm, y)

    def sample(
        self,
        T: float,
        model_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
        n_samples: int = 1,
        random_state: int = None,
    ) -> RenewalData:
        """Renewal data sampling.

        Parameters
        ----------
        T : float
            Time at the end of the observation.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().
        n_samples : int, optional
            Number of samples, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        RenewalData
            Samples of replacement times and durations.
        """
        n_indices = max(1, args_size(*model_args, *model1_args))
        if self.model1 is not None:
            data = self._sample_init(
                T, self.model1, model1_args, n_indices, n_samples, random_state
            )
        else:
            data = self._sample_init(
                T, self.model, model_args, n_indices, n_samples, random_state
            )
        return self._sample_loop(data, self.model, model_args)

    @classmethod
    def _sample_init(
        cls,
        T: float,
        model: LifetimeModel,
        model_args: Tuple[np.ndarray, ...],
        n_indices: int,
        n_samples: int,
        random_state: int = None,
    ) -> RenewalData:
        """Initializing a RenewalData sample.

        Creates a RenewalData instance with the first renewal times and durations.

        Parameters
        ----------
        T : float
            Time at the end of observation.
        model : LifetimeModel
            A lifetime model representing the durations between events.
        model_args : Tuple[ndarray,...]
            Extra arguments required by the lifetime model.
        n_indices : int
            Number of assets.
        n_samples : int
            Number of samples.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        RenewalData
            The renewal data sample with the first renewal times and durations.
        """
        indices = np.repeat(np.arange(n_indices), n_samples)
        samples = np.tile(np.arange(n_samples), n_indices)
        size = n_indices * n_samples if len(model_args) == 0 else 1
        times = model.rvs(
            *args_take(indices, *model_args), size=size, random_state=random_state
        ).ravel()
        durations = times.copy()
        return RenewalData(T, n_indices, n_samples, indices, samples, times, durations)

    @classmethod
    def _sample_loop(
        cls, data: RenewalData, model: LifetimeModel, model_args: Tuple[np.ndarray, ...]
    ) -> RenewalData:
        """Sampling loop on renewals until the end time is reached.

        Parameters
        ----------
        data : RenewalData
            An initial RenewalData instance.
        model : LifetimeModel
            A lifetime model representing the durations between events.
        model_args : Tuple[ndarray,...]
            Extra arguments required by the lifetime model.

        Returns
        -------
        RenewalData
            The renewal data sample with times and durations.
        """
        T = data.T
        ind_T = np.nonzero(data.times < T)
        times = data.times[ind_T]
        indices = data.indices[ind_T]
        samples = data.samples[ind_T]

        while len(ind_T[0])>0:
            size = indices.size if len(model_args) == 0 else 1
            durations = model.rvs(*args_take(indices, *model_args), size=size).ravel()
            times = times + durations
            data.times = np.concatenate((data.times, times))
            data.durations = np.concatenate((data.durations, durations))
            data.indices = np.concatenate((data.indices, indices))
            data.samples = np.concatenate((data.samples, samples))
            ind_T = np.nonzero(times < T)
            times = times[ind_T]
            indices = indices[ind_T]
            samples = samples[ind_T]

        return data


class RenewalRewardProcess(RenewalProcess):
    """Renewal reward process."""

    def __init__(
        self,
        model: LifetimeModel,
        reward: Reward,
        model1: LifetimeModel = None,
        reward1: Reward = None,
        discount: Discount = ExponentialDiscounting(),
    ):
        """Creates a renewal reward process.

        Parameters
        ----------
        model : LifetimeModel
            A lifetime model representing the durations between events.
        reward : Reward
            A reward associated to the interarrival time.
        model1 : LifetimeModel, optional
            A lifetime model for the first renewal (delayed renewal process), by
            default None.
        reward1 : Reward, optional
            A reward associated to the first renewal, by default None
        discount : Discount, optional
            A discount function related to the rewards, by default
            ExponentialDiscounting()
        """
        super().__init__(model, model1)
        self.reward = reward
        self.reward1 = reward1
        self.discount = discount

    def expected_total_reward(
        self,
        t: np.ndarray,
        model_args: Tuple[np.ndarray, ...] = (),
        reward_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
        reward1_args: Tuple[np.ndarray, ...] = (),
        discount_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        r"""The expected total reward.

        Parameters
        ----------
        t : 1D array
            Timeline.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        reward_args : Tuple[ndarray,...], optional
            Extra arguments required by the associated reward, by default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().
        reward1_args : Tuple[ndarray,...], optional
           Extra arguments required by the associated reward of the first renewal, by default ().
        discount_args : Tuple[ndarray,...], optional
            Extra arguments required by the discount function, by default ().

        Returns
        -------
        ndarray
            Expected total reward of process evaluated at each point of the
            timeline.

        Raises
        ------
        NotImplementedError
            If the discount function is not exponential.

        Notes
        -----
        The renewal equation solved by the expected reward is:

        .. math::

            z(t) = \int_0^t E[Y | X = x] D(x) \mathrm{d}F(x) + \int_0^t z(t-x)
            D(x)\mathrm{d}F(x)

        where:

        - :math:`z` is the expected total reward,
        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`X` the interarrival random variable,
        - :math:`Y` the associated reward,
        - :math:`D` the exponential discount factor.

        If the renewal reward process is delayed, the expected total reward is
        modified as:

        .. math::

            z_1(t) = \int_0^t E[Y_1 | X_1 = x] D(x) \mathrm{d}F_1(x) + \int_0^t
            z(t-x) D(x) \mathrm{d}F_1(x)

        where:

        - :math:`z_1` is the expected total reward with delay,
        - :math:`F_1` is the cumulative distribution function of the lifetime
          model for the first renewal,
        - :math:`X_1` the interarrival random variable of the first renewal,
        - :math:`Y_1` the associated reward of the first renewal,
        """

        _check_exponential_discounting(self.discount)
        tm = 0.5 * (t[1:] + t[:-1])
        F = self.model.cdf(t, *model_args)
        Fm = self.model.cdf(tm, *model_args)
        D = self.discount.factor(t, *discount_args)
        y = self._reward_partial_expectation(
            t,
            self.model,
            self.reward,
            self.discount,
            model_args,
            reward_args,
            discount_args,
        )
        z = renewal_equation_solver(F, Fm, y, D)
        if self.model1 is not None:
            F1 = self.model1.cdf(t, *model1_args)
            F1m = self.model1.cdf(tm, *model1_args)
            y1 = self._reward_partial_expectation(
                t,
                self.model1,
                self.reward1,
                self.discount,
                model1_args,
                reward1_args,
                discount_args,
            )
            z = delayed_renewal(z, F1, F1m, y1, D)
        return z

    def expected_equivalent_annual_worth(
        self,
        t: np.ndarray,
        model_args: Tuple[np.ndarray, ...] = (),
        reward_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
        reward1_args: Tuple[np.ndarray, ...] = (),
        discount_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        """Expected equivalent annual worth.

        Gives the equivalent annual worth of the expected total reward of the
        process at each point of the timeline.

        Parameters
        ----------
        t : 1D array
            Timeline.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        reward_args : Tuple[ndarray,...], optional
            Extra arguments required by the associated reward, by default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().
        reward1_args : Tuple[ndarray,...], optional
           Extra arguments required by the associated reward of the first
           renewal, by default ().
        discount_args : Tuple[ndarray,...], optional
            Extra arguments required by the discount function, by default ().

        Returns
        -------
        ndarray
            The expected equivalent annual worth evaluated at each point of the
            timeline.

        Notes
        -----
        The equivalent annual worth at time :math:`t` is equal to the expected
        total reward :math:`z` divided by the annuity factor :math:`AF(t)`.
        """
        z = self.expected_total_reward(
            t, model_args, reward_args, model1_args, reward1_args, discount_args
        )
        af = self.discount.annuity_factor(t, *discount_args)
        mask = af == 0
        af = np.ma.masked_where(mask, af)
        q = z / af
        if self.model1 is not None:
            q0 = self.reward1.conditional_expectation(
                0, *reward1_args
            ) * self.model1.pdf(0, *model1_args)
        else:
            q0 = self.reward.conditional_expectation(0, *reward_args) * self.model.pdf(
                0, *model_args
            )
        return np.where(mask, q0, q)

    def asymptotic_expected_total_reward(
        self,
        model_args: Tuple[np.ndarray, ...] = (),
        reward_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
        reward1_args: Tuple[np.ndarray, ...] = (),
        discount_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        r"""Asymptotic expected total reward.

        Parameters
        ----------
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        reward_args : Tuple[ndarray,...], optional
            Extra arguments required by the associated reward, by default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().
        reward1_args : Tuple[ndarray,...], optional
           Extra arguments required by the associated reward of the first
           renewal, by default ().
        discount_args : Tuple[ndarray,...], optional
            Extra arguments required by the discount function, by default ().

        Returns
        -------
        ndarray
            The assymptotic expected total reward of the process.

        Raises
        ------
        NotImplementedError
            If the discount function is not exponential.

        Notes
        -----
        The asymptotic expected total reward is:

        .. math::

            z^\infty = \lim_{t\to \infty} z(t) = \dfrac{E[Y D(X)]}{1-E[D(X)]}

        where:

        - :math:`X` the interarrival random variable,
        - :math:`Y` the associated reward,
        - :math:`D` the exponential discount factor.

        If the renewal reward process is delayed, the asymptotic expected total
        reward is modified as:

        .. math::

            z_1^\infty = E[Y_1 D(X_1)] + z^\infty E[D(X_1)]

        where:

        - :math:`X_1` the interarrival random variable of the first renewal,
        - :math:`Y_1` the associated reward of the first renewal,
        """
        _check_exponential_discounting(self.discount)
        rate = discount_args[0]
        mask = rate <= 0
        rate = np.ma.MaskedArray(rate, mask)
        ndim = args_ndim(*model_args, *reward_args, rate)
        f = lambda x: self.discount.factor(x, rate)
        y = lambda x: self.discount.factor(
            x, rate
        ) * self.reward.conditional_expectation(x, *reward_args)
        lf = self.model.ls_integrate(f, 0, np.inf, *model_args, ndim=ndim)
        ly = self.model.ls_integrate(y, 0, np.inf, *model_args, ndim=ndim)
        z = ly / (1 - lf)
        if self.model1 is not None:
            ndim1 = args_ndim(*model1_args, *reward1_args, rate)
            y1 = lambda x: self.discount.factor(
                x, rate
            ) * self.reward.conditional_expectation(x, *reward1_args)
            lf1 = self.model1.ls_integrate(f, 0, np.inf, *model1_args, ndim=ndim1)
            ly1 = self.model1.ls_integrate(y1, 0, np.inf, *model1_args, ndim=ndim1)
            z = ly1 + z * lf1
        return np.where(mask, np.inf, z)

    def asymptotic_expected_equivalent_annual_worth(
        self,
        model_args: Tuple[np.ndarray, ...] = (),
        reward_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
        reward1_args: Tuple[np.ndarray, ...] = (),
        discount_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        """Asymptotic expected equivalent annual worth.

        Parameters
        ----------
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        reward_args : Tuple[ndarray,...], optional
            Extra arguments required by the associated reward, by default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().
        reward1_args : Tuple[ndarray,...], optional
           Extra arguments required by the associated reward of the first
           renewal, by default ().
        discount_args : Tuple[ndarray,...], optional
            Extra arguments required by the discount function, by default ().

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual worth of the process.

        Raises
        ------
        NotImplementedError
            If the discount function is not exponential.
        """
        _check_exponential_discounting(self.discount)
        rate = discount_args[0]
        mask = rate <= 0
        rate = np.ma.MaskedArray(rate, mask)
        ndim = args_ndim(*model_args, *reward_args)
        q = rate * self.asymptotic_expected_total_reward(
            model_args, reward_args, model1_args, reward1_args, discount_args
        )
        q0 = self.model.ls_integrate(
            lambda x: self.reward.conditional_expectation(x, *reward_args),
            0,
            np.inf,
            *model_args,
            ndim=ndim
        ) / self.model.mean(*model_args)
        return np.where(mask, q0, q)

    def sample(
        self,
        T: float,
        model_args: Tuple[np.ndarray, ...] = (),
        reward_args: Tuple[np.ndarray, ...] = (),
        model1_args: Tuple[np.ndarray, ...] = (),
        reward1_args: Tuple[np.ndarray, ...] = (),
        discount_args: Tuple[np.ndarray, ...] = (),
        n_samples: int = 1,
        random_state: int = None,
    ) -> RenewalRewardData:
        """Renewal reward data sampling.

        Parameters
        ----------
        T : float
            Time at the end of the observation.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        reward_args : Tuple[ndarray,...], optional
            Extra arguments required by the associated reward, by default ().
        model1_args : Tuple[ndarray,...], optional
            Extra arguments required by the lifetime model of the first renewal,
            by default ().
        reward1_args : Tuple[ndarray,...], optional
           Extra arguments required by the associated reward of the first
           renewal, by default ().
        discount_args : Tuple[ndarray,...], optional
            Extra arguments required by the discount function, by default ().
        n_samples : int, optional
            Number of samples, by default 1.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        RenewalRewardData
            Samples of replacement times, durations and rewards.
        """
        n_indices = max(
            1, args_size(*model_args, *reward_args, *model1_args, *reward1_args)
        )
        if self.model1 is not None:
            data = self._sample_init(
                T,
                self.model1,
                self.reward1,
                self.discount,
                model1_args,
                reward1_args,
                discount_args,
                n_indices,
                n_samples,
                random_state,
            )
        else:
            data = self._sample_init(
                T,
                self.model,
                self.reward,
                self.discount,
                model_args,
                reward_args,
                discount_args,
                n_indices,
                n_samples,
                random_state,
            )
        return self._sample_loop(
            data,
            self.model,
            self.reward,
            self.discount,
            model_args,
            reward_args,
            discount_args,
        )

    @classmethod
    def _reward_partial_expectation(
        cls,
        t: np.ndarray,
        model: LifetimeModel,
        reward: Reward,
        discount: Discount,
        model_args: Tuple[np.ndarray, ...] = (),
        reward_args: Tuple[np.ndarray, ...] = (),
        discount_args: Tuple[np.ndarray, ...] = (),
    ) -> np.ndarray:
        r"""Reward partial expectation.

        Parameters
        ----------
        t : float or 1D array
            Timeline.
        model : LifetimeModel
            A lifetime model representing the durations between events.
        reward : Reward
            A reward associated to the interarrival time.
        discount : Discount
            A discount function related to the rewards.
        model_args : Tuple[ndarray,...], optional
            Extra arguments required by the underlying lifetime model, by
            default ().
        reward_args : Tuple[ndarray,...], optional
            Extra arguments required by the associated reward, by default ().
        discount_args : Tuple[ndarray,...], optional
            Extra arguments required by the discount function, by default ().

        Returns
        -------
        ndarray
            Reward partial expectation at each point of the timeline.

        Notes
        -----
        The reward partial expactation is defined by:

        .. math::

            y(t) = \int_0^t E[Y | X = x] D(x) \mathrm{d}F(x)

        where:

        - :math:`F` is the cumulative distribution function of the underlying
          lifetime model,
        - :math:`X` the interarrival random variable,
        - :math:`Y` the associated reward,
        - :math:`D` the exponential discount factor.
        """
        func = lambda x: reward.conditional_expectation(
            x, *reward_args
        ) * discount.factor(x, *discount_args)
        ndim = args_ndim(t, *model_args, *reward_args, *discount_args)
        return model.ls_integrate(func, 0, t, *model_args, ndim=ndim)

    @classmethod
    def _sample_init(
        cls,
        T: float,
        model: LifetimeModel,
        reward: Reward,
        discount: Discount,
        model_args: Tuple[np.ndarray, ...],
        reward_args: Tuple[np.ndarray, ...],
        discount_args: Tuple[np.ndarray, ...],
        n_indices: int,
        n_samples: int,
        random_state: int = None,
    ) -> RenewalRewardData:
        """Initializing a RenewalRewardData sample.

        Creates a RenewalRewardData instance with the first renewal times,
        durations and rewards.

        Parameters
        ----------
        T : float
            Time at the end of the observation.
        model : LifetimeModel
            A lifetime model representing the durations between events.
        reward : Reward
            A reward associated to the interarrival time.
        discount : Discount
            A discount function related to the rewards.
        model_args : Tuple[ndarray,...]
            Extra arguments required by the underlying lifetime model.
        reward_args : Tuple[ndarray,...]
            Extra arguments required by the associated reward.
        discount_args : Tuple[ndarray,...]
            Extra arguments required by the discount function.
        n_indices : int
            Number of assets.
        n_samples : int
            Number of samples.
        random_state : int, optional
            Random seed, by default None.

        Returns
        -------
        RenewalRewardData
            The renewal reward data sample with the first renewal times,
            durations and rewards.
        """
        data = super()._sample_init(
            T, model, model_args, n_indices, n_samples, random_state
        )
        rewards = (
            np.array(
                reward.sample(
                    data.durations.reshape(-1, 1),
                    *args_take(data.indices, *reward_args)
                ).swapaxes(-2, -1)
                * discount.factor(data.times, *discount_args),
                ndmin=3,
            )
            .sum(axis=0)
            .ravel()
        )
        return RenewalRewardData(*data.astuple(), rewards)

    @classmethod
    def _sample_loop(
        cls,
        data: RenewalRewardData,
        model: LifetimeModel,
        reward: Reward,
        discount: Discount,
        model_args: Tuple[np.ndarray, ...],
        reward_args: Tuple[np.ndarray, ...],
        discount_args: Tuple[np.ndarray, ...],
    ) -> RenewalRewardData:
        """Sampling loop on renewals until the end time is reached.

        Parameters
        ----------
        data : RenewalRewardData
            An initial RenewalRewardData instance.
        model : LifetimeModel
            A lifetime model representing the durations between events.
        reward : Reward
            A reward associated to the interarrival time.
        discount : Discount
            A discount function related to the rewards.
        model_args : Tuple[ndarray,...]
            Extra arguments required by the underlying lifetime model.
        reward_args : Tuple[ndarray,...]
            Extra arguments required by the associated reward.
        discount_args : Tuple[ndarray,...]
            Extra arguments required by the discount function.

        Returns
        -------
        RenewalRewardData
           The renewal reward data sample with times, durations and rewards.
        """
        data = super()._sample_loop(data, model, model_args)
        ind = data.rewards.size
        indices = data.indices[ind:]
        times = data.times[ind:]
        durations = data.durations[ind:]
        rewards = (
            np.array(
                reward.sample(
                    durations.reshape(-1, 1), *args_take(indices, *reward_args)
                ).swapaxes(-2, -1)
                * discount.factor(times, *discount_args),
                ndmin=3,
            )
            .sum(axis=0)
            .ravel()
        )
        data.rewards = np.concatenate((data.rewards, rewards))
        return data
