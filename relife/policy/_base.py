from abc import ABC, abstractmethod

import numpy as np

from relife.economic import ExponentialDiscounting


def _make_timeline(tf, nb_steps):
    timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
    return np.atleast_2d(timeline)  # (1, nb_steps) to ensure broadcasting


class _OneCycleExpectedCosts:

    def __init__(
        self,
        lifetime_model,
        reward,
        discounting_rate=0.0,
        period_before_discounting=1.0,
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
        timeline = np.atleast_2d(np.array(np.inf))  # (1, 1) to ensure broadcasting
        return np.squeeze(self._expected_equivalent_annual_cost(timeline)[-1])  # () or (m,)


class ReplacementPolicy(ABC):

    def __init__(
        self,
        baseline_model,
        cost_structure,
        discounting_rate=0.0,
    ):
        self.baseline_model = baseline_model
        self.discounting_rate = discounting_rate
        self._cost_structure = cost_structure  # hidden, contains reshaped cost arrays

    @abstractmethod
    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        r"""
        The expected net present value.

        The net present value is commonly computed with a time discrete formula.
        It has a continuous variation where cash flows are time dependant.
        From a random perspective, one can compute its expected value called :math:`z(t)`.

        .. math::

            z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_1 \sim F` is the random lifetime of the first asset
        - :math:`Z_t` are the random costs at each time :math:`t`
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
            A tuple containing the timeline and the computed values.
        """

    @abstractmethod
    def asymptotic_expected_net_present_value(self, total_sum=False):
        r"""
        The asymtotic expected net present value :math:`\lim_{t\to\infty} z(t)`.

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """

    @abstractmethod
    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        r"""
        The expected equivalent annual cost.

        The equivalent annual cost corresponds to the value of the constant cash flow that
        leads to the same net present value. It is generally used to compare investment projects of unequal lifespans.
        It has a continuous variation and from a random perspective, one can compute its expected value called :math:`q(t)`.

        .. math::

            q(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time.
        - :math:`z(t)` is the expected net present value at time :math:`t`.
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
            A tuple containing the timeline and the computed values.
        """

    @abstractmethod
    def asymptotic_expected_equivalent_annual_cost(self, total_sum=False):
        r"""
        The asymtotic expected equivalent annual cost, :math:`\lim_{t\to\infty} q(t)`.

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the total sum over the first axis of the result. If the policy data encodes several
            assets, this option allows to return the sum result on the flit rather than calling ``np.sum`` afterwards.

        Returns
        -------
        ndarray
            The asymptotic expected values.
        """
