import numpy as np
from jedi.inference.recursion import total_function_execution_limit

from relife.economic import RunToFailureReward
from relife.lifetime_model import LeftTruncatedModel
from relife.stochastic_process import RenewalRewardProcess
from relife.utils import reshape_1d_arg, is_lifetime_model

from ._base import _OneCycleExpectedCosts

def run_to_failure_policy(model, cf, one_cycle=False, **kwargs):
    if is_lifetime_model(model):
        if one_cycle:
            return OneCycleRunToFailurePolicy(model, cf, **kwargs)
        return RunToFailurePolicy(model, cf, **kwargs)
    else:
        raise ValueError("can't create a preventive age replacement policy from given model")

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

    def __init__(self, lifetime_model, cf, discounting_rate=0.0, period_before_discounting=1.0, a0=None):
        self.lifetime_model = lifetime_model
        self._cf = reshape_1d_arg(cf)
        self._a0 = reshape_1d_arg(a0) if a0 is not None else a0
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
                discounting_rate=self.discounting_rate,
                period_before_discounting=self.period_before_discounting,
            )
        return _OneCycleExpectedCosts(
            LeftTruncatedModel(self.lifetime_model).freeze_args(self.a0),
            RunToFailureReward(self.cf),
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
        )

    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        r"""
        Calculate the expected net present value over a given timeline.

        It takes into account ``discounting_rate`` attribute value.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost
        total_sum : bool, default False
            If True, returns the sum of every expected equivalent annual cost.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.
        """
        timeline, npv = self._expected_costs.expected_net_present_value(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            return timeline, np.sum(npv, axis=0)
        return timeline, npv

    def asymptotic_expected_net_present_value(self, total_sum=False):
        r"""
        Calculate the asymptotic net present value.

        It takes into account ``discounting_rate`` attribute value.

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the sum of every asymptotic net present value.

        Returns
        -------
        np.ndarray
            The asymptotic expected total cost.
        """
        asymptotic_npv = self._expected_costs.asymptotic_expected_net_present_value()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        r"""
        Calculate the expected equivalent annual cost over a given timeline.

        It takes into account ``discounting_rate`` attribute value.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost
        total_sum : bool, default False
            If True, returns the sum of every expected equivalent annual cost.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected total cost and its corresponding values at each
            step of the timeline.
        """
        timeline, eeac = self._expected_costs.expected_equivalent_annual_cost(tf, nb_steps)
        if total_sum and eeac.ndim == 2:
            return timeline, np.sum(eeac, axis=0)
        return timeline, eeac

    def asymptotic_expected_equivalent_annual_cost(self, total_sum=False):
        r"""
        Calculate the asymptotic expected equivalent annual cost.

        It takes into account ``discounting_rate`` attribute value.

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the sum of every asymptotic expected equivalent annual cost.

        Returns
        -------
        np.ndarray
            The asymptotic expected total cost.
        """
        asymptotic_eeac = self._expected_costs.asymptotic_expected_equivalent_annual_cost()
        if total_sum:
            return np.sum(asymptotic_eeac)
        return asymptotic_eeac


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
        self._cf = reshape_1d_arg(cf)
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
                discounting_rate=self.discounting_rate,
            )
        return RenewalRewardProcess(
            self.lifetime_model,
            RunToFailureReward(self.cf),
            discounting_rate=self.discounting_rate,
            first_lifetime_model=LeftTruncatedModel(self.lifetime_model).freeze_args(self.a0),
        )

    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        r"""
        The expected net present value.

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
        total_sum : bool, default False
            If True, returns the sum of every expected net present value.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected net present value and its corresponding values at each
            step of the timeline.
        """
        timeline, npv = self._stochastic_process.expected_total_reward(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            npv = np.sum(npv, axis=0)
        return timeline, npv

    def asymptotic_net_present_value(self, total_sum=False):
        r"""
        The asymtotic expected total cost

        .. math::

            \lim_{t\to\infty} z(t)

        where :math:`z(t)` is the expected total cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_net_present_value` for more details.

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the sum of every asymptotic net present value.

        Returns
        -------
        ndarray
            The asymptotic expected total cost values
        """
        asymptotic_npv = self._stochastic_process.asymptotic_expected_total_reward()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        r"""
        The expected equivalent annual cost.

        .. math::

            \text{EEAC}(t) = \dfrac{\delta z(t)}{1 - e^{-\delta t}}

        where :

        - :math:`t` is the time
        - :math:`z(t)` is the expected_net_present_value at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_net_present_value` for more details.`.
        - :math:`\delta` is the discounting rate.

        Parameters
        ----------
        tf : float
            Time horizon. The expected equivalent annual cost will be computed up until this calendar time.
        nb_steps : int
            The number of steps used to compute the expected equivalent annual cost
        total_sum : bool, default False
            If True, returns the sum of every expected equivalent annual cost.

        Returns
        -------
        tuple of two ndarrays
            A tuple containing the timeline used to compute the expected annual cost and its corresponding values at each
            step of the timeline.
        """
        timeline, eeac = self._stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)
        if total_sum and eeac.ndim == 2:
            eeac  = np.sum(eeac, axis=0)
        return timeline, eeac

    def asymptotic_expected_equivalent_annual_cost(self, total_sum=False):
        r"""
        The asymtotic expected equivalent annual cost

        .. math::

            \lim_{t\to\infty} \text{EEAC}(t)

        where :math:`\text{EEAC}(t)` is the expected equivalent annual cost at :math:`t`. See :py:meth:`~AgeReplacementPolicy.expected_equivalent_annual_cost` for more details.

        Parameters
        ----------
        total_sum : bool, default False
            If True, returns the sum of every asymptotic expected equivalent annual cost.

        Returns
        -------
        ndarray
            The asymptotic expected equivalent annual cost
        """
        asymptotic_eeac = self._stochastic_process.asymptotic_expected_equivalent_annual_worth()
        if total_sum:
            return np.sum(asymptotic_eeac)
        return asymptotic_eeac

    def sample(self, size, tf, t0=0.0, seed=None):
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        size : int
            The size of the desired sample.
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        size : int or tuple of 2 int
            Size of the sample
        seed : int, optional
            Random seed, by default None.

        """

        return self._stochastic_process.sample(tf, t0, size, seed)
