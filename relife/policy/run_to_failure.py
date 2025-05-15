from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from relife.economic import RunToFailureReward
from relife.stochastic_process import RenewalRewardProcess

from ._base import RenewalPolicy

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        LifetimeDistribution,
    )


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
    """

    model1 = None

    def __init__(
        self,
        model: LifetimeDistribution | FrozenParametricLifetimeModel,
        cf: float | NDArray[np.float64],
        discounting_rate: float = 0.0,
        period_before_discounting: float = 1.0,
    ) -> None:
        super().__init__(model, discounting_rate=discounting_rate, cf=cf)
        if period_before_discounting <= 0:
            raise ValueError("The period_before_discounting must be greater than 0")
        self.period_before_discounting = period_before_discounting
        self.reward = RunToFailureReward(cf)

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @property
    def cf(self):
        return self.cost["cf"]

    def expected_total_cost(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)
        # reward partial expectation
        return self.model.ls_integrate(
            lambda x: self.reward.sample(x) * self.discounting.factor(x), np.zeros_like(timeline), timeline, deg=10
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        # reward partial expectation
        return self.model.ls_integrate(
            lambda x: self.reward.sample(x) * self.discounting.factor(x), 0.0, np.inf, deg=10
        )

    def _expected_equivalent_annual_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        def f(x: float | NDArray[np.float64]) -> np.float64 | NDArray[np.float64]:
            return (
                self.reward.conditional_expectation(x) * self.discounting.factor(x) / self.discounting.annuity_factor(x)
            )

        q0 = self.model.cdf(self.period_before_discounting) * f(self.period_before_discounting)  # () or (m, 1)
        a = np.full_like(timeline, self.period_before_discounting)  # (nb_steps,)
        # change first value of lower bound to compute the integral
        a = np.where(timeline < self.period_before_discounting, timeline, a)  # (nb_steps,)
        integral = np.atleast_2d(
            self.model.ls_integrate(f, a, timeline)
        )  # (nb_steps,) if q0: (), or (m, nb_steps) if q0 : (m, 1)
        mask = np.broadcast_to(
            timeline < self.period_before_discounting, integral.shape
        )  # (nb_steps,) or (m, nb_steps)
        q0 = np.broadcast_to(q0, integral.shape)  # (nb_steps,) or (m, nb_steps)
        integral = np.where(mask, q0, integral)
        return integral

    def expected_equivalent_annual_cost(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        return self._expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(
        self,
    ) -> NDArray[np.float64]:
        return self._expected_equivalent_annual_cost(np.array(np.inf))


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
        model: LifetimeDistribution | FrozenParametricLifetimeModel,
        cf: float | NDArray[np.float64],
        *,
        discounting_rate: float = 0.0,
        model1: Optional[LifetimeDistribution | FrozenParametricLifetimeModel] = None,
    ) -> None:
        super().__init__(model, model1, discounting_rate, cf=cf)

        self.reward = RunToFailureReward(cf)
        self.underlying_process = RenewalRewardProcess(
            self.model,
            self.reward,
            discounting_rate=self.discounting_rate,
            model1=self.model1,
        )

    @property
    def discounting_rate(self):
        return self.discounting.rate

    @property
    def cf(self):
        return self.cost["cf"]

    def expected_nb_replacements(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.underlying_process.renewal_function(tf, nb_steps)

    def expected_total_cost(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.underlying_process.expected_total_reward(tf, nb_steps)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, tf: float, nb_steps: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.underlying_process.expected_equivalent_annual_worth(tf, nb_steps)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.underlying_process.asymptotic_expected_equivalent_annual_worth()


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

DefaultRunToFailurePolicy.expected_total_cost.__doc__ = ETC_DOCSTRING
DefaultRunToFailurePolicy.expected_equivalent_annual_cost.__doc__ = EEAC_DOCSTRING
DefaultRunToFailurePolicy.asymptotic_expected_total_cost.__doc__ = ASYMPTOTIC_ETC_DOCSTRING
DefaultRunToFailurePolicy.asymptotic_expected_equivalent_annual_cost.__doc__ = ASYMPTOTIC_EEAC_DOCSTRING
