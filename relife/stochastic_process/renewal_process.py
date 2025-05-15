from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, NamedTuple, Optional

import numpy as np
from numpy.typing import DTypeLike, NDArray

from relife import ParametricModel
from relife.data import LifetimeData
from relife.economic import (
    Discounting,
    ExponentialDiscounting,
    reward_partial_expectation,
)
from relife.lifetime_model import FrozenParametricLifetimeModel, LifetimeDistribution

from ._renewal_equation import delayed_renewal_equation_solver, renewal_equation_solver

if TYPE_CHECKING:
    from relife.economic import Reward

    from .sample import SampleFunction


class CountData(NamedTuple):
    t0: float
    tf: float
    struct: NDArray[DTypeLike]  # struct array


def concatenate_count_data(
    model: RenewalProcess,
    tf: float,
    t0: float = 0.0,
    size: int | tuple[int] | tuple[int, int] = 1,
    maxsample: int = 1e5,
    seed: Optional[int] = None,
) -> CountData:
    from .sample import RenewalProcessIterator

    iterator = RenewalProcessIterator(model, size, (t0, tf), seed=seed)
    struct_arr = next(iterator)
    for arr in iterator:
        if len(arr) > maxsample:
            raise RuntimeError("Max number of sample reached")
        struct_arr = np.concatenate((struct_arr, arr))
    struct_arr = np.sort(struct_arr, order=("sample_id", "asset_id", "timeline"))
    return CountData(t0, tf, struct_arr)


class RenewalProcess(ParametricModel):

    sample_data: Optional[CountData]

    def __init__(
        self,
        model: LifetimeDistribution | FrozenParametricLifetimeModel,
        model1: Optional[LifetimeDistribution | FrozenParametricLifetimeModel] = None,
    ):
        super().__init__()

        self.model = model
        self.model1 = model1
        self.sample_data = None

    @property
    def sample(self) -> Optional[SampleFunction]:
        if self.sample_data is not None:
            from .sample import SampleFunction

            return SampleFunction(type(self), self.sample_data)

    def renewal_function(self, tf: float, nb_steps: int) -> NDArray[np.float64]:  # (nb_steps,) or (m, nb_steps)
        return renewal_equation_solver(
            tf,
            nb_steps,
            self.model,
            self.model.cdf if self.model1 is None else self.model1.cdf,
        )

    def renewal_density(self, tf: float, nb_steps: int) -> NDArray[np.float64]:  # (nb_steps,) or (m, nb_steps)
        return renewal_equation_solver(
            tf,
            nb_steps,
            self.model,
            self.model.pdf if self.model1 is None else self.model1.pdf,
        )

    def sample_count_data(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ):
        self.sample_data = concatenate_count_data(self, tf, t0, size, maxsample, seed)

    def sample_lifetime_data(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> LifetimeData:
        if self.model1 is not None and self.model1 != self.model:
            from relife.lifetime_model import LeftTruncatedModel

            if isinstance(self.model1, LeftTruncatedModel) and self.model1.baseline == self.model:
                pass
            else:
                raise ValueError(
                    f"Calling sample_failure_data on RenewalProcess having different model and model1 is ambiguous. Instantiate RenewalProcess with only one model"
                )

        count_data = concatenate_count_data(self, tf, t0, size, maxsample, seed)
        return LifetimeData(
            count_data["data"]["time"].copy(),
            event=count_data["data"]["event"].copy(),
            entry=count_data["data"]["entry"].copy(),
            args=tuple((np.take(arg, count_data.struct["asset_id"]) for arg in getattr(self.model, "frozen_args", ()))),
        )


class RenewalRewardProcess(RenewalProcess):

    discounting: Discounting

    def __init__(
        self,
        model: LifetimeDistribution | FrozenParametricLifetimeModel,
        reward: Reward,
        discounting_rate: float = 0.,
        model1: Optional[LifetimeDistribution | FrozenParametricLifetimeModel] = None,
        reward1: Optional[Reward] = None,
    ):
        super().__init__(model, model1)
        self.reward = reward
        self.reward1 = reward1 if reward1 is not None else reward
        self.discounting = ExponentialDiscounting(discounting_rate)

    def expected_total_reward(self, tf: float, nb_steps: int) -> NDArray[np.float64]:  # (nb_steps,) or (m, nb_steps)
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
        lf = self.model.ls_integrate(lambda x : self.discounting.factor(x), 0., np.inf, deg=10)
        ly = self.model.ls_integrate(lambda x: self.discounting.factor(x) * self.reward.sample(x), 0., np.inf, deg=10)
        z = ly / (1 - lf)
        if self.model1 is not None:
            lf1 = self.model1.ls_integrate(lambda x : self.discounting.factor(x), 0., np.inf, deg=10)
            ly1 = self.model1.ls_integrate(lambda x : self.discounting.factor(x) * self.reward1.sample(x),0., np.inf, deg=10)
            z = ly1 + z * lf1
        return z

    def expected_equivalent_annual_worth(self, tf: float, nb_steps: int) -> NDArray[np.float64]:
        z = self.expected_total_reward(tf, nb_steps)
        af = self.discounting.annuity_factor(np.linspace(0, tf, nb_steps, dtype=np.float64))
        q = z / (af + 1e-5)  # avoid zero division
        res = np.full_like(af, q)
        if self.model1 is None:
            q0 = self.reward.sample(0.) * self.model.pdf(0.)
        else:
            q0 = self.reward1.sample(0.) * self.model1.pdf(0.)
        res[af == 0.0] = q0
        return res

    def asymptotic_expected_equivalent_annual_worth(self) -> NDArray[np.float64]:
        if self.discounting.rate == 0.0:
            return np.squeeze(
                self.model.ls_integrate(
                    lambda x: self.reward.sample(x),
                    np.array(0.0),
                    np.array(np.inf),
                )
                / self.model.mean()
            )
        else:
            return self.discounting.rate * self.asymptotic_expected_total_reward()


# def total_rewards(count_data : CountData) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     sort = np.argsort(self.timeline)
#     timeline = self.timeline[sort]
#     rewards = self.rewards[sort]
#     timeline = np.insert(timeline, 0, self.t0)
#     rewards = np.insert(rewards, 0, 0)
#     rewards[timeline == self.tf] = 0
#     return timeline, rewards.cumsum()
#
#
# def mean_total_rewards(count_data : CountData) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
#     timeline, rewards = self.total_rewards()
#     return timeline, rewards / len(self)
