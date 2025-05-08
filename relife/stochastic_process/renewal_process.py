from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife import ParametricModel
from relife.economic import (
    Discounting,
    ExponentialDiscounting,
    reward_partial_expectation,
)
from relife.sample import CountDataSampleMixin, FailureDataSampleMixin

from ._renewal_equation import delayed_renewal_equation_solver, renewal_equation_solver

if TYPE_CHECKING:
    from relife.economic import Reward
    from relife.lifetime_model import (
        ParametricLifetimeModel,
    )


class RenewalProcess(ParametricModel, CountDataSampleMixin, FailureDataSampleMixin):
    def __init__(
        self,
        model: ParametricLifetimeModel[()],
        model1: Optional[ParametricLifetimeModel[()]] = None,
    ):
        super().__init__()

        if len(model.args_names) > 0:
            raise ValueError(
                "Invalid type of model. It must be ParametricLifetimeModel[()] object type. You may call freeze first"
            )
        if model1 is not None:
            if len(model1.args_names) > 0:
                raise ValueError(
                    "Invalid type of model1. It must be ParametricLifetimeModel[()] object type. You may call freeze first"
                )

        self.model = model
        self.model1 = model1


    def renewal_function(
        self, tf: float, nb_steps: int
    ) -> NDArray[np.float64]:  # (nb_steps,) or (m, nb_steps)
        return renewal_equation_solver(
            tf,
            nb_steps,
            self.model,
            self.model.cdf if self.model1 is None else self.model1.cdf,
        )

    def renewal_density(
        self, tf: float, nb_steps: int
    ) -> NDArray[np.float64]:  # (nb_steps,) or (m, nb_steps)
        return renewal_equation_solver(
            tf,
            nb_steps,
            self.model,
            self.model.pdf if self.model1 is None else self.model1.pdf,
        )


class RenewalRewardProcess(RenewalProcess):

    discounting: Discounting

    def __init__(
        self,
        model: ParametricLifetimeModel[()],
        reward: Reward,
        discounting_rate: Optional[float] = None,
        *,
        model1: Optional[ParametricLifetimeModel[()]] = None,
        reward1: Optional[Reward] = None,
    ):
        super().__init__(model, model1)
        self.reward = reward
        self.reward1 = reward1 if reward1 is not None else reward
        self.discounting = ExponentialDiscounting(discounting_rate)

    def expected_total_reward(
        self, tf: float, nb_steps: int
    ) -> NDArray[np.float64]:  # (nb_steps,) or (m, nb_steps)
        z = renewal_equation_solver(
            tf,
            nb_steps,
            self.model,
            partial(
                reward_partial_expectation,
                model=self.model,
                rewards=self.reward,
                discounting=self.discounting,
            ),
            discounting=self.discounting,
        )

        if self.model1 is None:
            return z
        else:
            return delayed_renewal_equation_solver(
                tf,
                nb_steps,
                z,
                self.model1,
                partial(
                    reward_partial_expectation,
                    model=self.model1,
                    rewards=self.reward1,
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
            return self.discounting.factor(x) * self.reward(x)

        lf = self.model.ls_integrate(f, np.array(0.0), np.array(np.inf))
        ly = self.model.ls_integrate(y, np.array(0.0), np.array(np.inf))
        z = ly / (1 - lf)

        if self.model1 is not None:

            def y1(x):
                return self.discounting.factor(x) * self.reward1(x)

            lf1 = self.model1.ls_integrate(f, np.array(0.0), np.array(np.inf))
            ly1 = self.model1.ls_integrate(y1, np.array(0.0), np.array(np.inf))
            z = ly1 + z * lf1

        return np.squeeze(z)

    def expected_equivalent_annual_worth(
        self, tf: float, nb_steps: int
    ) -> NDArray[np.float64]:
        z = self.expected_total_reward(tf, nb_steps)
        af = self.discounting.annuity_factor(
            np.linspace(0, tf, nb_steps, dtype=np.float64)
        )
        q = z / (af + 1e-5)  # avoid zero division
        res = np.full_like(af, q)
        if self.model1 is None:
            q0 = self.reward(np.array(0.0)) * self.model.pdf(0.0)
        else:
            q0 = self.reward1(np.array(0.0)) * self.model1.pdf(0.0)
        res[af == 0.0] = q0

        return res

    def asymptotic_expected_equivalent_annual_worth(self) -> NDArray[np.float64]:
        if self.discounting.rate == 0.0:
            return np.squeeze(
                self.model.ls_integrate(
                    lambda x: self.reward(x),
                    np.array(0.0),
                    np.array(np.inf),
                )
                / self.model.mean()
            )
        else:
            return self.discounting.rate * self.asymptotic_expected_total_reward()