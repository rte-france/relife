import numpy as np

from relife.economic import RunToFailureReward
from relife.lifetime_model import LeftTruncatedModel
from relife.stochastic_process import RenewalRewardProcess
from relife.utils import flatten_if_possible, is_lifetime_model, reshape_1d_arg

from ._base import ReplacementPolicy, _OneCycleExpectedCosts


def run_to_failure_policy(baseline_model, cf, one_cycle=False, **kwargs):
    if is_lifetime_model(baseline_model):
        if one_cycle:
            return OneCycleRunToFailurePolicy(baseline_model, cf, **kwargs)
        return RunToFailurePolicy(baseline_model, cf, **kwargs)
    else:
        raise ValueError("can't create a run-to-failure policy from the given model")


class OneCycleRunToFailurePolicy(ReplacementPolicy):
    r"""One cyle run-to-failure policy.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    current_ages : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.
    """

    def __init__(self, lifetime_model, cf, discounting_rate=0.0, current_ages=None, period_before_discounting=1.0):
        super().__init__(lifetime_model, cost_structure={"cf": reshape_1d_arg(cf)}, discounting_rate=discounting_rate)
        self._current_ages = reshape_1d_arg(current_ages) if current_ages is not None else current_ages
        self.period_before_discounting = period_before_discounting

    @property
    def current_ages(self):
        """Current ages of the assets.

        Returns
        -------
        np.ndarray
        """
        # _a0 is (m, 1) but exposed cf is (m,)
        if self._current_ages is None:
            return self._current_ages
        return flatten_if_possible(self._current_ages)

    @property
    def cf(self):
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return flatten_if_possible(self._cost_structure["cf"])

    @cf.setter
    def cf(self, value):
        self._cost_structure["cf"] = reshape_1d_arg(value)

    @property
    def _expected_costs(self):
        if self.current_ages is None:
            return _OneCycleExpectedCosts(
                self.baseline_model,
                RunToFailureReward(self.cf),
                discounting_rate=self.discounting_rate,
                period_before_discounting=self.period_before_discounting,
            )
        return _OneCycleExpectedCosts(
            LeftTruncatedModel(self.baseline_model).freeze_args(self.current_ages),
            RunToFailureReward(self.cf),
            discounting_rate=self.discounting_rate,
            period_before_discounting=self.period_before_discounting,
        )

    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        timeline, npv = self._expected_costs.expected_net_present_value(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            return timeline, np.sum(npv, axis=0)
        return timeline, npv

    def asymptotic_expected_net_present_value(self, total_sum=False):
        asymptotic_npv = self._expected_costs.asymptotic_expected_net_present_value()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        timeline, eeac = self._expected_costs.expected_equivalent_annual_cost(tf, nb_steps)
        if total_sum and eeac.ndim == 2:
            return timeline, np.sum(eeac, axis=0)
        return timeline, eeac

    def asymptotic_expected_equivalent_annual_cost(self, total_sum=False):
        asymptotic_eeac = self._expected_costs.asymptotic_expected_equivalent_annual_cost()
        if total_sum:
            return np.sum(asymptotic_eeac)
        return asymptotic_eeac


class RunToFailurePolicy(ReplacementPolicy):
    r"""Run-to-failure renewal policy.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    current_ages : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    def __init__(self, lifetime_model, cf, discounting_rate=0.0, current_ages=None):
        super().__init__(lifetime_model, cost_structure={"cf": reshape_1d_arg(cf)}, discounting_rate=discounting_rate)
        self._current_ages = reshape_1d_arg(current_ages) if current_ages is not None else current_ages

    @property
    def current_ages(self):
        """Current ages of the assets.

        Returns
        -------
        np.ndarray
        """
        # _a0 is (m, 1) but exposed cf is (m,)
        if self._current_ages is None:
            return self._current_ages
        return flatten_if_possible(self._current_ages)

    @property
    def _stochastic_process(self):
        if self.current_ages is None:
            return RenewalRewardProcess(
                self.baseline_model,
                RunToFailureReward(self.cf),
                discounting_rate=self.discounting_rate,
            )
        return RenewalRewardProcess(
            self.baseline_model,
            RunToFailureReward(self.cf),
            discounting_rate=self.discounting_rate,
            first_lifetime_model=LeftTruncatedModel(self.baseline_model).freeze_args(self.current_ages),
        )

    @property
    def cf(self):
        """Cost of failure.

        Returns
        -------
        np.ndarray
        """
        # _cf is (m, 1) but exposed cf is (m,)
        return flatten_if_possible(self._cost_structure["cf"])

    @cf.setter
    def cf(self, value):
        self._cost_structure["cf"] = reshape_1d_arg(value)

    def expected_net_present_value(self, tf, nb_steps, total_sum=False):
        timeline, npv = self._stochastic_process.expected_total_reward(tf, nb_steps)
        if total_sum and npv.ndim == 2:
            npv = np.sum(npv, axis=0)
        return timeline, npv

    def asymptotic_expected_net_present_value(self, total_sum=False):
        asymptotic_npv = self._stochastic_process.asymptotic_expected_total_reward()
        if total_sum:
            return np.sum(asymptotic_npv)
        return asymptotic_npv

    def expected_equivalent_annual_cost(self, tf, nb_steps, total_sum=False):
        timeline, eeac = self._stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)
        if total_sum and eeac.ndim == 2:
            eeac = np.sum(eeac, axis=0)
        return timeline, eeac

    def asymptotic_expected_equivalent_annual_cost(self, total_sum=False):
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
