from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from relife.economic.rewards import run_to_failure_rewards
from relife.parametric_model import LeftTruncatedModel
from relife.stochastic_process import RenewalRewardProcess
from relife.stochastic_process.renewal import reward_partial_expectation
from .renewal import RenewalPolicy

if TYPE_CHECKING:
    from relife.model import BaseLifetimeModel


class OneCycleRunToFailurePolicy(RenewalPolicy):
    r"""One cyle run-to-failure policy

    A policy for running assets to failure within one cycle.

    Parameters
    ----------
    model : BaseLifetimeModel
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    period_before_discounting: float, default is 1.
        The length of the first period before discounting.
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    """

    model1 = None

    def __init__(
        self,
        model: BaseLifetimeModel,
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        period_before_discounting: float = 1.0,
        a0: Optional[float | NDArray[np.float64]] = None,
    ) -> None:
        super().__init__(model, discounting_rate=discounting_rate, cf=cf)
        if period_before_discounting == 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting

        if a0 is not None:
            self.model = LeftTruncatedModel(self.model).freeze(a0)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @property
    def cf(self):
        return self.cost_structure["cf"]

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            run_to_failure_rewards(self.cf),
            discounting=self.discounting,
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        f = (
            lambda x: run_to_failure_rewards(self.cf)(x)
            * self.discounting.factor(x)
            / self.discounting.annuity_factor(x)
        )
        mask = timeline < self.period_before_discounting
        q0 = self.model.cdf(self.period_before_discounting) * f(
            self.period_before_discounting
        )
        return np.squeeze(
            q0
            + np.where(
                mask,
                0,
                self.model.ls_integrate(f, self.period_before_discounting, timeline),
            )
        )

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf))


class DefaultRunToFailurePolicy(RenewalPolicy):
    r"""Run-to-failure renewal policy.

    Renewal reward stochastic_process where assets are replaced on failure with costs
    :math:`c_f`.

    Parameters
    ----------
    model : BaseLifetimeModel
        The lifetime model of the assets.
    cf : np.ndarray
        The cost of failure for each asset.
    discounting_rate : float, default is 0.
        The discounting rate.
    a0 : ndarray, optional
        Current ages of the assets (default is None). Setting ``a0`` will add
        left truncations.
    model1 : Lifetimemodel, optional
        The lifetime model used for the first cycle of replacements. When one adds
        `model1`, we assume that `model1` is different from `model` meaning
        the underlying survival probabilities behave differently for the first
        cycle.


    References
    ----------
    .. [1] Van der Weide, J. A. M., & Van Noortwijk, J. M. (2008). Renewal
        theory with exponential and hyperbolic discounting. Probability in
        the Engineering and Informational Sciences, 22(1), 53-74.
    """

    def __init__(
        self,
        model: BaseLifetimeModel[()],
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: Optional[float] = None,
        a0: Optional[float | NDArray[np.float64]] = None,
        model1: Optional[BaseLifetimeModel[()]] = None,
    ) -> None:
        super().__init__(model, model1, discounting_rate, cf=cf)

        if a0 is not None:
            if self.model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            self.model1 = LeftTruncatedModel(self.model).freeze(a0)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @property
    def cf(self):
        return self.cost_structure["cf"]

    @property
    def underlying_process(
        self,
    ) -> RenewalRewardProcess:
        return RenewalRewardProcess(
            self.model,
            run_to_failure_rewards(self.cf),
            discounting_rate=self.discounting_rate,
            model1=self.model1,
            rewards1=run_to_failure_rewards(self.cf) if self.model1 else None,
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.underlying_process.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.underlying_process.expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_equivalent_annual_cost()


from ._docstring import (
    ASYMPTOTIC_EEAC_DOCSTRING,
    ASYMPTOTIC_ETC_DOCSTRING,
    EEAC_DOCSTRING,
    ETC_DOCSTRING,
)

OneCycleRunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
OneCycleRunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
OneCycleRunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = (
    ASYMPTOTIC_ETC_DOCSTRING
)
OneCycleRunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING
)

DefaultRunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
DefaultRunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
DefaultRunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = (
    ASYMPTOTIC_ETC_DOCSTRING
)
DefaultRunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = (
    ASYMPTOTIC_EEAC_DOCSTRING
)
