from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife import ParametricModel
from relife.economic import exponential_discounting, reward_partial_expectation

from ._renewal_equation import delayed_renewal_equation_solver, renewal_equation_solver

if TYPE_CHECKING:
    from relife.economic import Rewards
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        ParametricLifetimeModel,
    )
    from relife.sample import CountData


class RenewalProcess(ParametricModel):
    model: FrozenParametricLifetimeModel
    model1: Optional[FrozenParametricLifetimeModel]

    def __init__(
        self,
        model: ParametricLifetimeModel[()],
        model1: Optional[ParametricLifetimeModel[()]] = None,
    ):
        super().__init__()

        from relife.lifetime_model import LifetimeDistribution

        if not model.frozen:
            raise ValueError(
                "Invalid model : must be Lifetimemodel[()] object. You may call freeze_zvariables first"
            )
        if isinstance(model, LifetimeDistribution):
            model = model.freeze()
        self.compose_with(model=model)
        if model1 is not None:
            if not model1.frozen:
                raise ValueError(
                    "Invalid model1 : must be Lifetimemodel[()] object. You may call freeze_zvariables first"
                )
            if not isinstance(model1, LifetimeDistribution):
                model1 = model1.freeze()
            self.compose_with(model1=model1)

    def renewal_function(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return renewal_equation_solver(
            timeline,
            self.model,
            self.model.cdf if not self.model1 else self.model1.cdf,
        )

    def renewal_density(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return renewal_equation_solver(
            timeline,
            self.model,
            self.model.pdf if not self.model1 else self.model1.pdf,
        )

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sample import sample_count_data

        return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)

    def failure_data_sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sample import failure_data_sample

        return failure_data_sample(
            self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use="model"
        )


class RenewalRewardProcess(RenewalProcess):

    def __init__(
        self,
        model: ParametricLifetimeModel[()],
        rewards: Rewards,
        discounting_rate: Optional[float] = None,
        *,
        model1: Optional[ParametricLifetimeModel[()]] = None,
        rewards1: Optional[Rewards] = None,
    ):
        super().__init__(model, model1)
        self.rewards = rewards
        self.rewards1 = rewards1 if rewards1 is not None else rewards
        self.discounting = exponential_discounting(discounting_rate)

    def expected_total_reward(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        z = renewal_equation_solver(
            timeline,
            self.model,
            partial(
                reward_partial_expectation,
                model=self.model,
                rewards=self.rewards,
                discounting=self.discounting,
            ),
            discounting=self.discounting,
        )

        if self.model1 is None:
            return z
        else:
            return delayed_renewal_equation_solver(
                timeline,
                z,
                self.model1,
                partial(
                    reward_partial_expectation,
                    model=self.model1,
                    rewards=self.rewards1,
                    discounting=self.discounting,
                ),
                discounting=self.discounting,
            )

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        def f(x):
            return self.discounting.factor(x)

        def y(x):
            return self.discounting.factor(x) * self.rewards(x)

        lf = self.model.ls_integrate(f, np.array(0.0), np.array(np.inf))
        ly = self.model.ls_integrate(y, np.array(0.0), np.array(np.inf))
        z = ly / (1 - lf)

        if self.model1 is not None:

            def y1(x):
                return self.discounting.factor(x) * self.rewards1(x)

            lf1 = self.model1.ls_integrate(f, np.array(0.0), np.array(np.inf))
            ly1 = self.model1.ls_integrate(y1, np.array(0.0), np.array(np.inf))
            z = ly1 + z * lf1

        return np.squeeze(z)

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        z = self.expected_total_reward(timeline)
        af = self.discounting.annuity_factor(timeline)
        q = z / af
        res = np.full_like(af, q)
        if self.model1 is None:
            q0 = self.rewards(np.array(0.0)) * self.model.pdf(0.0)
        else:
            q0 = self.rewards1(np.array(0.0)) * self.model1.pdf(0.0)
        res[af == 0.0] = q0

        return res

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        if self.discounting.rate == 0.0:
            return np.squeeze(
                self.model.ls_integrate(
                    lambda x: self.rewards(x),
                    np.array(0.0),
                    np.array(np.inf),
                )
                / self.model.mean()
            )
        else:
            return self.discounting.rate * self.asymptotic_expected_total_reward()

    @override
    def failure_data_sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
        use: str = "model",
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sample import failure_data_sample

        return failure_data_sample(
            self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use=use
        )
