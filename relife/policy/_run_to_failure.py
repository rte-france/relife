import numpy as np

from relife.utils import to_relife_shape
from relife.economic import RunToFailureReward, ExponentialDiscounting
from relife.lifetime_model import LeftTruncatedModel
from relife.stochastic_process import RenewalRewardProcess

from ._base import _OneCycleExpectedCosts

class OneCycleRunToFailurePolicy:
    r"""One cyle run-to-failure policy.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    a0 : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.
    """

    def __init__(self, lifetime_model, cf, discounting_rate = 0.0, period_before_discounting = 1.0, a0 = None):
        self.lifetime_model = lifetime_model
        self._cf = to_relife_shape(cf)
        self._a0 = to_relife_shape(a0) if a0 is not None else a0
        self.discounting_rate = discounting_rate
        self.period_before_discounting = period_before_discounting

    @property
    def cf(self):
        """Costs of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return np.squeeze(self._cf)

    @property
    def a0(self):
        """Current ages of the assets.

        Returns
        -------
        np.ndarray
        """
        if self._a0 is None:
            return self._a0
        return np.squeeze(self._a0)

    @property
    def _expected_costs(self):
        if self.a0 is None:
            return _OneCycleExpectedCosts(
                self.lifetime_model,
                RunToFailureReward(self.cf),
                ExponentialDiscounting(self.discounting_rate),
                self.period_before_discounting
            )
        return _OneCycleExpectedCosts(
            LeftTruncatedModel(self.lifetime_model).freeze_args(self.a0),
        RunToFailureReward(self.cf),
        ExponentialDiscounting(self.discounting_rate),
        self.period_before_discounting
        )

    def expected_total_cost(self, tf, nb_steps):
        r"""
        Calculate the expected total cost over a given timeline.

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
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        return self._expected_costs.expected_total_cost(tf, nb_steps)

    def asymptotic_expected_total_cost(self):
        r"""
        Calculate the asymptotic expected total cost.

        It takes into account ``discounting_rate`` attribute value.

        Returns
        -------
        np.ndarray
            The asymptotic expected total cost.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        return self._expected_costs.asymptotic_expected_total_cost()

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
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        return self._expected_costs.expected_equivalent_annual_cost(tf, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self):
        r"""
        Calculate the asymptotic expected equivalent annual cost.

        It takes into account ``discounting_rate`` attribute value.

        Returns
        -------
        np.ndarray
            The asymptotic expected total cost.

        .. warning::

            This method requires the ``ar`` attribute to be set either at initialization
            or with the ``optimize`` method.
        """
        return self._expected_costs.asymptotic_expected_equivalent_annual_cost()


class RunToFailurePolicy:
    r"""Run-to-failure renewal policy.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    a0 : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    def __init__(
        self,
        lifetime_model,
        cf,
        discounting_rate=0.0,
        a0=None,
    ):
        self.lifetime_model = lifetime_model
        self._cf = to_relife_shape(cf)
        self._a0 = to_relife_shape(a0) if a0 is not None else a0
        self.discounting_rate = discounting_rate

    @property
    def cf(self):
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return np.squeeze(self._cf)

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
        return np.squeeze(self._a0)

    @property
    def _stochastic_process(self):
        if self.a0 is None:
            return RenewalRewardProcess(
                self.lifetime_model,
                RunToFailureReward(self.cf),
                ExponentialDiscounting(self.discounting_rate),
            )
        return RenewalRewardProcess(
            self.lifetime_model,
            RunToFailureReward(self.cf),
            ExponentialDiscounting(self.discounting_rate),
            first_lifetime_model=LeftTruncatedModel(self.lifetime_model).freeze_args(self.a0)
        )

    def expected_total_cost(self, tf, nb_steps):
        r"""
        The expected total cost.

        It is computed by solving the renewal equation and is given by:

        .. math::

            z(t) = \mathbb{E}(Z_t) = \int_{0}^{\infty}\mathbb{E}(Z_t~|~X_1 = x)dF(x)

        where :

        - :math:`t` is the time
        - :math:`X_i \sim F` are :math:`n` random variable lifetimes, *i.i.d.*, of cumulative distribution :math:`F`.
        - :math:`Z_t` is the random variable reward at each time :math:`t`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            Time horizon. The expected total cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected total cost

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.
        """
        return self._stochastic_process.expected_total_reward(tf, nb_steps)

    def asymptotic_expected_total_cost(self):
        r"""
        The asymtotic expected total cost

        .. math::

            \lim_{t\to\infty} z(t)

        where :math:`z(t)` is the expected total cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_total_cost` for more details.

        Returns
        -------
        ndarray
            The asymptotic expected total cost values
        """
        return self._stochastic_process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(self, tf, nb_steps):
        r"""
        The expected equivalent annual cost.

        .. math::

            \text{EEAC}(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time
        - :math:`z(t)` is the expected_total_cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_total_cost` for more details.`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected annual cost and its corresponding values at each
            step of the timeline.
        """
        return self._stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self):
        r"""
        The asymtotic expected equivalent annual cost

        .. math::

            \lim_{t\to\infty} \text{EEAC}(t)

        where :math:`\text{EEAC}(t)` is the expected equivalent annual cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_equivalent_annual_cost` for more details.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost
        """
        return self._stochastic_process.asymptotic_expected_equivalent_annual_worth()