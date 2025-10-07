import numpy as np

from relife.utils import get_args_nb_assets
from relife.economic import RunToFailureReward, ExponentialDiscounting
from relife.lifetime_model import LeftTruncatedModel
from relife.stochastic_process import RenewalRewardProcess

from ._base import _OneCycleExpectedCosts

def _reshape_policy_data(value, nb_assets):
    value = np.squeeze(np.asarray(value))
    if nb_assets > 1:
        value = np.broadcast_to(value, (nb_assets, 1)).copy()
    return value

class OneCycleRunToFailurePolicy:
    # noinspection PyUnresolvedReferences
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

    Attributes
    ----------
    cf
    """

    def __init__(self, lifetime_model, cf, discounting_rate = 0.0, period_before_discounting = 1.0, a0 = None):
        self.lifetime_model = lifetime_model
        self._nb_assets = get_args_nb_assets(cf, a0, *getattr(lifetime_model, "args", ()))
        self._cf = _reshape_policy_data(cf, self._nb_assets)
        self._a0 = _reshape_policy_data(a0, self._nb_assets) if a0 is not None else a0
        self.discounting_rate = discounting_rate
        self.period_before_discounting = period_before_discounting

    @property
    def cf(self):
        # _cf is (m, 1) but exposed cf is (m,)
        return np.squeeze(self._cf)

    @property
    def a0(self):
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
        return self._expected_costs.expected_total_cost(tf, nb_steps)

    def asymptotic_expected_total_cost(self):
        return self._expected_costs.asymptotic_expected_total_cost()

    def expected_equivalent_annual_cost(self, tf, nb_steps):
        return self._expected_costs.expected_equivalent_annual_cost(tf, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self):
        return self._expected_costs.asymptotic_expected_equivalent_annual_cost()


class RunToFailurePolicy:
    # noinspection PyUnresolvedReferences
    r"""Run-to-failure renewal policy.

    Renewal reward stochastic_process where assets are replaced on failure with costs
    :math:`c_f`.

    Parameters
    ----------
    lifetime_model : any lifetime distribution or frozen lifetime model
        A lifetime model representing the durations between events.
    cf : float or 1darray
        Costs of failures
    discounting_rate : float, default is 0.
        The discounting rate value used in the exponential discounting function
    first_lifetime_model : any lifetime distribution or frozen lifetime model, optional
        A lifetime model for the first renewal (delayed renewal process). It is None by default
    a0 : float or 1darray, optional
        Current ages of the assets, by default 0 for each asset. If it is given, left truncations of ``a0`` will
        be take into account for the first cycle.

    Attributes
    ----------
    cf

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
        self._nb_assets = get_args_nb_assets(cf, a0, *getattr(lifetime_model, "args", ()))
        self._cf = _reshape_policy_data(cf, self._nb_assets)
        self._a0 = _reshape_policy_data(a0, self._nb_assets) if a0 is not None else a0
        self.discounting_rate = discounting_rate

    @property
    def cf(self):
        # _cf is (m, 1) but exposed cf is (m,)
        return np.squeeze(self._cf)

    @property
    def a0(self):
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
        return self._stochastic_process.expected_total_reward(tf, nb_steps)

    def asymptotic_expected_total_cost(self):
        return self._stochastic_process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(self, tf, nb_steps):
        return self._stochastic_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self):
        return self._stochastic_process.asymptotic_expected_equivalent_annual_worth()