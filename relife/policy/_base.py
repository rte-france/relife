import numpy as np

from relife.economic import ExponentialDiscounting


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

    def _make_timeline(self, tf: float, nb_steps: int):
        # tile is necessary to ensure broadcasting of the operations
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        args_nb_assets = getattr(self.lifetime_model, "nb_assets", 1)  # default 1 for LifetimeDistribution case
        if args_nb_assets > 1:
            timeline = np.tile(timeline, (args_nb_assets, 1))
        elif self.reward.ndim == 2:  # elif because we consider that if m > 1 in frozen_model, in reward it is 1 or m
            timeline = np.tile(timeline, (self.reward.size, 1))
        return timeline  # (nb_steps,) or (m, nb_steps)

    def expected_net_present_value(self, tf: float, nb_steps: int):
        r"""
        Calculate the expected net present value over a given timeline.

        It takes into account ``discounting_rate`` attribute value.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected net present value and its corresponding values at each
            step of the timeline.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        timeline = self._make_timeline(tf, nb_steps)
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
        r"""
        Calculate the asymptotic expected net present value.

        It takes into account ``discounting_rate`` attribute value.

        Returns
        -------
        np.ndarray
            The asymptotic expected net present value.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
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
        r"""
        Calculate the expected equivalent annual cost over a given timeline.

        It takes into account ``discounting_rate`` attribute value.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected net present value and its corresponding values at each
            step of the timeline.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        timeline = self._make_timeline(tf, nb_steps)  # (nb_steps,) or (m, nb_steps)
        return self._expected_equivalent_annual_cost(timeline)  # (nb_steps,) or (m, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self):
        r"""
        Calculate the asymptotic expected equivalent annual cost.

        It takes into account ``discounting_rate`` attribute value.

        Returns
        -------
        np.ndarray
            The asymptotic expected net present value.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        timeline = np.array(np.inf)
        args_nb_assets = getattr(self.lifetime_model, "args_nb_assets", 1)  #  default 1 for LifetimeDistribution case
        if args_nb_assets > 1:
            timeline = np.tile(timeline, (args_nb_assets, 1))
        elif self.reward.ndim == 2:  # elif because we consider that if m > 1 in frozen_model, in reward it is 1 or m
            timeline = np.tile(timeline, (self.reward.size, 1))
        # timeline : () or (m, 1)
        return np.squeeze(self._expected_equivalent_annual_cost(timeline)[-1])  # () or (m,)
