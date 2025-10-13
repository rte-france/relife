from abc import ABC, abstractmethod

import numpy as np

from relife.economic import ExponentialDiscounting
from relife.utils import reshape_1d_arg, flatten_if_possible


def _make_timeline(tf, nb_steps):
    timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
    return np.atleast_2d(timeline)  # (1, nb_steps) to ensure broadcasting


class _OneCycleExpectedCosts:

    def __init__(
        self,
        lifetime_model,
        reward,
        discounting_rate = 0.,
        period_before_discounting = 1.0,
    ) -> None:
        self.reward = reward
        self.discounting = ExponentialDiscounting(discounting_rate)
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.lifetime_model = lifetime_model

    def expected_net_present_value(self, tf, nb_steps):
        timeline = _make_timeline(tf, nb_steps)
        etc = self.lifetime_model.ls_integrate(
            lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x),
            np.zeros_like(timeline),
            timeline,
            deg=15,
        )  # (nb_steps,) or (m, nb_steps)
        if timeline.ndim == 2:
            return timeline[0, :], etc  # (nb_steps,) and (m, nb_steps)
        return timeline, etc  # (nb_steps,) and (nb_steps,)

    def asymptotic_expected_net_present_value(self):
        # reward partial expectation
        return np.squeeze(
            self.lifetime_model.ls_integrate(
                lambda x: self.reward.conditional_expectation(x) * self.discounting.factor(x), 0.0, np.inf, deg=15
            )
        )  # () or (m,)

    def _expected_equivalent_annual_cost(self, timeline):

        # timeline : (nb_steps,) or (m, nb_steps)
        def f(x):
            # avoid zero division + 1e-6
            return (
                self.reward.conditional_expectation(x)
                * self.discounting.factor(x)
                / (self.discounting.annuity_factor(x) + 1e-6)
            )

        q0 = self.lifetime_model.cdf(self.period_before_discounting) * f(self.period_before_discounting)  # () or (m, 1)
        a = np.full_like(timeline, self.period_before_discounting)  # (nb_steps,) or (m, nb_steps)

        # change first value of lower bound to compute the integral
        a[timeline < self.period_before_discounting] = 0.0  # (nb_steps,)
        # a = np.where(timeline < self.period_before_discounting, 0., a)  # (nb_steps,)
        integral = self.lifetime_model.ls_integrate(
            f, a, timeline, deg=15
        )  # (nb_steps,) or (m, nb_steps) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (), (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.where(mask, q0, q0 + integral)
        if timeline.ndim == 2:
            return timeline[0, :], np.squeeze(integral)  # (nb_steps,) and (m, nb_steps)
        return timeline, np.squeeze(integral)  # (nb_steps,) and (nb_steps,)

    def expected_equivalent_annual_cost(self, tf, nb_steps):
        timeline = _make_timeline(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        return self._expected_equivalent_annual_cost(timeline)  # (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self):
        timeline = np.atleast_2d(np.array(np.inf)) # (1, 1) to ensure broadcasting
        return np.squeeze(self._expected_equivalent_annual_cost(timeline)[-1])  # () or (m,)


class ReplacementPolicy(ABC):

    def __init__(
        self,
        lifetime_model,
        cf,
        discounting_rate=0.0,
        a0=None,
    ):
        self.lifetime_model = lifetime_model
        self.cf = cf
        self._a0 = reshape_1d_arg(a0) if a0 is not None else a0
        self.discounting_rate = discounting_rate

    @property
    def cf(self):
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return flatten_if_possible(self._cf)

    @cf.setter
    def cf(self, value):
        self._cf = reshape_1d_arg(value)

    @property
    def a0(self):
        """Current ages of the assets.

        Returns
        -------
        np.ndarray
        """
        # _a0 is (m, 1) but exposed cf is (m,)
        if self._a0 is None:
            return self._a0
        return flatten_if_possible(self._a0)

    @abstractmethod
    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        r"""
        The expected net present value.

        .. math::

            \text{NPV}(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_1 \sim F` is lifetime of the first cycle
        - :math:`Z_t` are the random rewards
        - :math:`\delta` is the discounting rate

        It is computed by solving the renewal equation.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.
        """

    @abstractmethod
    def asymptotic_expected_net_present_value(self, total_sum=False):
        r"""
        The asymtotic expected net present value.

        .. math::

            \lim_{t\to\infty} \text{NPV}(t)

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected net present value.
        """

    @abstractmethod
    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        r"""
        The expected equivalent annual cost.

        .. math::

            \text{EEAC}(t) = \dfrac{\delta \text{NPV}(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time.
        - :math:`\text{NPV}(t)` is the net present value at time :math:`t`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            The final time.
        nb_steps : int
            The number of steps used to discretized the time.
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected annual cost and its corresponding values at each
            step of the timeline.
        """

    @abstractmethod
    def asymptotic_expected_equivalent_annual_cost(self, total_sum=False):
        r"""
        The asymtotic expected equivalent annual cost.

        .. math::

            \lim_{t\to\infty} \text{EEAC}(t)

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost.
        """