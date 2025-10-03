from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.economic import RunToFailureReward
from relife.lifetime_model import LeftTruncatedModel
from relife.stochastic_process import RenewalRewardProcess

from ._base import BaseAgeReplacementPolicy, BaseOneCycleAgeReplacementPolicy

class OneCycleRunToFailurePolicy(BaseOneCycleAgeReplacementPolicy):
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

    reward: RunToFailureReward

    def __init__(
        self,
        lifetime_model,
        cf: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
        a0: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        reward = RunToFailureReward(cf)
        if a0 is not None:
            lifetime_model = LeftTruncatedModel(lifetime_model).freeze_args(a0)
        super().__init__(lifetime_model, reward, discounting_rate, period_before_discounting)

    @property
    def cf(self):
        return self.reward.cf


class RunToFailurePolicy(BaseAgeReplacementPolicy):
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
        discounting_rate = 0.0,
        first_lifetime_model = None,
        a0 = None,
    ) -> None:

        reward = RunToFailureReward(cf)
        first_reward = None
        if first_lifetime_model is not None:
            if a0 is not None:
                first_lifetime_model = LeftTruncatedModel(first_lifetime_model).freeze_args(a0)
            first_reward = RunToFailureReward(cf)
        elif a0 is not None:
            first_lifetime_model = LeftTruncatedModel(lifetime_model).freeze_args(a0)

        stochastic_process = RenewalRewardProcess(
            lifetime_model,
            reward,
            discounting_rate,
            first_lifetime_model=first_lifetime_model,
            first_reward=first_reward,
        )
        super().__init__(stochastic_process)

    @property
    def cf(self):
        """
        Cost of failures

        Returns
        -------
        ndarray
        """
        return self.stochastic_process.reward.cf

    @cf.setter
    def cf(self, value):
        self.stochastic_process.reward.cf = value
        self.stochastic_process.first_reward.cf = value
