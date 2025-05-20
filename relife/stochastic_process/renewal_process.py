from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

import numpy as np
from numpy.typing import NDArray

from relife import ParametricModel, args_nb_assets
from relife.data import LifetimeData
from relife.economic import (
    Discounting,
    ExponentialDiscounting,
)
from relife.lifetime_model import (
    FrozenAgeReplacementModel,
    FrozenLeftTruncatedModel,
    FrozenLifetimeRegression,
    LifetimeDistribution,
)

if TYPE_CHECKING:
    from relife.economic import Reward

    from .sample import CountData, SampleFunction


class RenewalProcess(ParametricModel):

    sample_data: Optional[CountData]

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel,
        first_lifetime_model: Optional[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel
        ] = None,
    ):
        super().__init__()

        self.lifetime_model = lifetime_model
        self.first_lifetime_model = first_lifetime_model
        self.sample_data = None

    @property
    def sample(self) -> Optional[SampleFunction]:
        if self.sample_data is not None:
            from .sample import SampleFunction
            return SampleFunction(type(self), self.sample_data)
        return None

    def renewal_function(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        # timeline needs to be reshaped to ensure broadcasting
        if args_nb_assets(self.lifetime_model) > 1:
            timeline = np.tile(timeline, (args_nb_assets(self.lifetime_model), 1))
        # timeline : (nb_steps,) or (m, nb_steps)
        return timeline, renewal_equation_solver(
            timeline,
            self.lifetime_model,
            self.lifetime_model.cdf if self.first_lifetime_model is None else self.first_lifetime_model.cdf,
        )

    def renewal_density(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        # timeline needs to be reshaped to ensure broadcasting
        if args_nb_assets(self.lifetime_model) > 1:
            timeline = np.tile(timeline, (args_nb_assets(self.lifetime_model), 1))
        # timeline : (nb_steps,) or (m, nb_steps)
        return timeline, renewal_equation_solver(
            timeline,
            self.lifetime_model,
            self.lifetime_model.pdf if self.first_lifetime_model is None else self.first_lifetime_model.pdf,
        )

    def sample_count_data(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> None:
        from .sample import concatenate_count_data

        self.sample_data = concatenate_count_data(self, tf, t0, size, maxsample, seed)

    def sample_lifetime_data(
        self,
        tf: float,
        t0: float = 0.0,
        size: int | tuple[int] | tuple[int, int] = 1,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> LifetimeData:
        from .sample import concatenate_count_data

        if self.first_lifetime_model is not None and self.first_lifetime_model != self.lifetime_model:
            from relife.lifetime_model import FrozenLeftTruncatedModel

            if (
                isinstance(self.first_lifetime_model, FrozenLeftTruncatedModel)
                and self.first_lifetime_model.unfreeze() == self.lifetime_model
            ):
                pass
            else:
                raise ValueError(
                    f"Calling sample_failure_data on RenewalProcess having different model and model1 is ambiguous. Instantiate RenewalProcess with only one model"
                )

        count_data = concatenate_count_data(self, tf, t0, size, maxsample, seed)
        return LifetimeData(
            count_data.struct["time"].copy(),
            event=count_data.struct["event"].copy(),
            entry=count_data.struct["entry"].copy(),
            args=tuple(
                (np.take(arg, count_data.struct["asset_id"]) for arg in getattr(self.lifetime_model, "frozen_args", ()))
            ),
        )


class RenewalRewardProcess(RenewalProcess):

    def __init__(
        self,
        lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel,
        reward: Reward,
        discounting_rate: float = 0.,
        first_lifetime_model: Optional[
            LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel
        ] = None,
        first_reward: Optional[Reward] = None,
    ):
        super().__init__(lifetime_model, first_lifetime_model)
        self.reward = reward
        self.first_reward = first_reward if first_reward is not None else reward
        self.discounting = ExponentialDiscounting(discounting_rate)


    def expected_total_reward(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:  # (nb_steps,) or (m, nb_steps)
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        # timeline needs to be reshaped to ensure broadcasting
        if self.reward.cost_array.size > 1:
            # HERE timeline needs a reshape to ensure broadcasting in the func evaluation inside ls_integrate
            # eg : if lifetime_model .pdf return (nb_steps,) but reward is (m, 1), then a, b bounds will be (deg, nb_steps)
            # => reward.sample = (deg, nb_steps) * (m, 1) => broadcasting ERROR
            # timeline must be (m, nb_steps) so that a, b are (deg, m, nb_steps)
            # ls_integrate will be (m, nb_steps)
            timeline = np.tile(timeline, (self.reward.cost_array.size, 1))
        elif args_nb_assets(self.lifetime_model) > 1: # elif because we consider that if m > 1 in reward, it is the same in lifetime_model
            timeline = np.tile(timeline, (args_nb_assets(self.lifetime_model), 1))

        z = renewal_equation_solver(
            timeline,
            self.lifetime_model,
            lambda timeline: self.lifetime_model.ls_integrate(
                lambda x: self.reward.sample(x) * self.discounting.factor(x), np.zeros_like(timeline), timeline, deg=15
            ),  # reward partial expectation
            discounting=self.discounting,
        )
        if self.first_lifetime_model is not None:
            return delayed_renewal_equation_solver(
                timeline,
                z,
                self.first_lifetime_model,
                lambda timeline: self.first_lifetime_model.ls_integrate(
                    lambda x: self.first_reward.sample(x) * self.discounting.factor(x),
                    np.zeros_like(timeline),
                    timeline,
                    deg=15,
                ),  # reward partial expectation
                discounting=self.discounting,
            )
        return timeline, z

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        lf = self.lifetime_model.ls_integrate(lambda x: self.discounting.factor(x), 0.0, np.inf, deg=15)
        ly = self.lifetime_model.ls_integrate(
            lambda x: self.discounting.factor(x) * self.reward.sample(x), 0.0, np.inf, deg=15
        )
        z = ly / (1 - lf) # () or (m, 1)
        if self.first_lifetime_model is not None:
            lf1 = self.first_lifetime_model.ls_integrate(lambda x: self.discounting.factor(x), 0.0, np.inf, deg=15)
            ly1 = self.first_lifetime_model.ls_integrate(
                lambda x: self.discounting.factor(x) * self.first_reward.sample(x), 0.0, np.inf, deg=15
            )
            z = ly1 + z * lf1  # () or (m, 1)
        return z  # () or (m, 1)

    def expected_equivalent_annual_worth(self, tf: float, nb_steps: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        z = self.expected_total_reward(tf, nb_steps) # (nb_steps,) or (m, nb_steps)
        timeline = np.linspace(0, tf, nb_steps, dtype=np.float64)  # (nb_steps,)
        if self.reward.cost_array.size > 1:
            timeline = np.tile(timeline, (self.reward.cost_array.size, 1))
        elif args_nb_assets(self.lifetime_model) > 1:
            timeline = np.tile(timeline, (args_nb_assets(self.lifetime_model), 1))
        # timeline : (nb_steps,) or (m, nb_steps)
        af = self.discounting.annuity_factor(timeline) # (nb_steps,) or (m, nb_steps)
        q = z / (af + 1e-5) # (nb_steps,) or (m, nb_steps) # avoid zero division
        if self.first_lifetime_model is not None:
            q0 = self.first_reward.sample(0.0) * self.first_lifetime_model.pdf(0.0)
        else:
            q0 = self.reward.sample(0.0) * self.lifetime_model.pdf(0.0)
        # q0 : () or (m, 1)
        q0 = np.broadcast_to(q0, af.shape) # (), (nb_steps,) or (m, nb_steps)
        return timeline, np.where(af == 0, q0, q)

    def asymptotic_expected_equivalent_annual_worth(self) -> NDArray[np.float64]:
        if self.discounting.rate == 0.0:
            return (
                self.lifetime_model.ls_integrate(lambda x: self.reward.sample(x), 0.0, np.inf, deg=15)
                / self.lifetime_model.mean()
            ) # () or (m, 1)
        return self.discounting.rate * self.asymptotic_expected_total_reward() # () or (m, 1)


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


# replace t by timeline
def renewal_equation_solver(
    timeline : NDArray[np.float64],
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel,
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:

    # timeline : (nb_steps,) or (m, nb_steps)
    tm = 0.5 * (timeline[..., 1:] + timeline[..., :-1])  # (nb_steps - 1,) or (m, nb_steps - 1)
    f = lifetime_model.cdf(timeline)  # (nb_steps,) or (m, nb_steps)
    fm = lifetime_model.cdf(tm)  # (nb_steps - 1,) or (m, nb_steps - 1)
    y = evaluated_func(timeline)  # (nb_steps,) or (m, nb_steps)

    if y.shape != f.shape:
        raise ValueError("Invalid shape between model and evaluated_func")

    if discounting is not None:
        d = discounting.factor(timeline) # (nb_steps,) or (m, nb_steps)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[..., 1:] - fm, 0, 1, axis=-1)
    v = d[..., :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[..., 0] * fm[..., 0])
    z[..., 0] = y[..., 0]
    z[..., 1] = q0 * (y[..., 1] + z[..., 0] * u[..., 1])
    for n in range(2, f.shape[-1]):
        z[..., n] = q0 * (y[..., n] + z[..., 0] * u[..., n] + np.sum(z[..., 1:n][..., ::-1] * v[..., 1:n], axis=-1))
    return z


def delayed_renewal_equation_solver(
    timeline : NDArray[np.float64],
    z: NDArray[np.float64],
    first_lifetime_model: (
        LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel
    ),
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    discounting: Optional[ExponentialDiscounting] = None,
) -> NDArray[np.float64]:
    # timeline : (nb_steps,) or (m, nb_steps)
    tm = 0.5 * (timeline[..., 1:] + timeline[..., :-1])  # (nb_steps - 1,) or (m, nb_steps - 1)
    f1 = first_lifetime_model.cdf(timeline)  # (nb_steps,) or (m, nb_steps)
    f1m = first_lifetime_model.cdf(tm)  # (nb_steps - 1,) or (m, nb_steps - 1)
    y1 = evaluated_func(timeline)  # (nb_steps,) or (m, nb_steps - 1)
    if discounting is not None:
        d = discounting.factor(timeline) # (nb_steps,) or (m, nb_steps - 1)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[..., 1:] - f1m, 0, 1, axis=-1)
    v1 = d[..., :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[..., 0] = y1[..., 0]
    z1[..., 1] = y1[..., 1] + z[..., 0] * u1[..., 1] + z[..., 1] * d[..., 0] * f1m[..., 0]
    for n in range(2, f1.shape[-1]):
        z1[..., n] = (
            y1[..., n]
            + z[..., 0] * u1[..., n]
            + z[..., n] * d[..., 0] * f1m[..., 0]
            + np.sum(z[..., 1:n][..., ::-1] * v1[..., 1:n], axis=-1)
        )
    return z1
