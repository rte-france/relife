from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from relife.economic import RunToFailureReward, ExponentialDiscounting
from ._base import OneCycleAgeRenewalPolicy, AgeRenewalPolicy

if TYPE_CHECKING:
    from relife.lifetime_model import LifetimeDistribution, FrozenLifetimeRegression, FrozenLeftTruncatedModel


class OneCycleRunToFailurePolicy(OneCycleAgeRenewalPolicy):

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel,
        cf : float | NDArray[np.float64],
        discounting_rate: float,
        period_before_discounting: float = 1.0,
    ) -> None:
        reward = RunToFailureReward(cf)
        discounting = ExponentialDiscounting(discounting_rate)
        super().__init__(lifetime_model, reward, discounting, period_before_discounting)

    @property
    def cf(self):
        return self.reward.cost["cf"]


class RunToFailurePolicy(AgeRenewalPolicy):
    r"""Run-to-failure renewal policy.

    Renewal reward stochastic_process where assets are replaced on failure with costs
    :math:`c_f`.

    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
        cf : float | NDArray[np.float64],
        discounting_rate: float,
        first_lifetime_model : Optional[LifetimeDistribution | FrozenLifetimeRegression | FrozenLeftTruncatedModel] = None,
    ) -> None:
        reward = RunToFailureReward(cf)
        discounting = ExponentialDiscounting(discounting_rate)
        first_reward = None
        if first_lifetime_model is not None:
            first_reward = RunToFailureReward(cf)
        super().__init__(lifetime_model, reward, discounting, first_lifetime_model, first_reward)

    @property
    def cf(self):
        return self.reward.cost["cf"]


from ._docstring import (
    ASYMPTOTIC_EEAC_DOCSTRING,
    ASYMPTOTIC_ETC_DOCSTRING,
    EEAC_DOCSTRING,
    ETC_DOCSTRING,
)

OneCycleRunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
OneCycleRunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
OneCycleRunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING
OneCycleRunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING

RunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
RunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
RunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING
RunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING
