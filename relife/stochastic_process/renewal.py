from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Callable, NewType, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.economic.discounting import exponential_discounting
from relife.model import BaseDistribution
from relife.sample import SampleFailureDataMixin, SampleMixin

if TYPE_CHECKING:
    from relife.economic.discounting import Discounting
    from relife.model import FrozenLifetimeModel, LifetimeModel


def renewal_equation_solver(
    timeline: NDArray[np.float64],
    model: LifetimeModel[()],
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    *,
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f = model.cdf(timeline)
    fm = model.cdf(tm)
    y = evaluated_func(timeline)
    if discounting is not None:
        d = discounting.factor(timeline)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[..., 1:] - fm, 0, 1, axis=-1)
    v = d[..., :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[..., 0] * fm[..., 0])
    z[..., 0] = y[..., 0]
    z[..., 1] = q0 * (y[..., 1] + z[..., 0] * u[..., 1])
    for n in range(2, f.shape[-1]):
        z[..., n] = q0 * (
            y[..., n]
            + z[..., 0] * u[..., n]
            + np.sum(z[..., 1:n][..., ::-1] * v[..., 1:n], axis=-1)
        )
    return z


def delayed_renewal_equation_solver(
    timeline: NDArray[np.float64],
    z: NDArray[np.float64],
    model1: LifetimeModel[()],
    evaluated_func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    discounting: Optional[Discounting] = None,
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f1 = model1.cdf(timeline)
    f1m = model1.cdf(tm)
    y1 = evaluated_func(timeline)

    if discounting is not None:
        d = discounting.factor(timeline)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[..., 1:] - f1m, 0, 1, axis=-1)
    v1 = d[..., :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[..., 0] = y1[..., 0]
    z1[..., 1] = (
        y1[..., 1] + z[..., 0] * u1[..., 1] + z[..., 1] * d[..., 0] * f1m[..., 0]
    )
    for n in range(2, f1.shape[-1]):
        z1[..., n] = (
            y1[..., n]
            + z[..., 0] * u1[..., n]
            + z[..., n] * d[..., 0] * f1m[..., 0]
            + np.sum(z[..., 1:n][..., ::-1] * v1[..., 1:n], axis=-1)
        )
    return z1


class RenewalProcess(SampleMixin[()], SampleFailureDataMixin[()]):
    model: FrozenLifetimeModel
    model1: Optional[FrozenLifetimeModel]

    def __init__(
        self,
        model: LifetimeModel[()],
        model1: Optional[LifetimeModel[()]] = None,
    ):

        if not model.frozen:
            raise ValueError(
                "Invalid model : must be Lifetimemodel[()] object. You may call freeze_zvariables first"
            )
        if not isinstance(model, BaseDistribution):
            model = model.freeze()

        self.model = model
        if model1 is not None:
            if not model1.frozen:
                raise ValueError(
                    "Invalid model1 : must be Lifetimemodel[()] object. You may call freeze_zvariables first"
                )
            if not isinstance(model1, BaseDistribution):
                model1 = model1.freeze()
        self.model1 = model1

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

    # def sample(
    #     self,
    #     size: int,
    #     tf: float,
    #     t0: float = 0.0,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> CountData:
    #     from relife.sample import sample_count_data
    #
    #     return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)
    #
    # def failure_data_sample(
    #     self,
    #     size: int,
    #     tf: float,
    #     t0: float = 0.0,
    #     maxsample: int = 1e5,
    #     seed: Optional[int] = None,
    # ) -> tuple[NDArray[np.float64], ...]:
    #     from relife.sample import failure_data_sample
    #
    #     return failure_data_sample(
    #         self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use="model"
    #     )


Rewards = NewType(
    "Rewards",
    Callable[[NDArray[np.float64]], NDArray[np.float64]],
)


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel[()],
    rewards: Rewards,
    *,
    discounting: Optional[Discounting] = None,
) -> np.ndarray:
    def func(x):
        return rewards(x) * discounting.factor(x)

    ls = model.ls_integrate(func, np.zeros_like(timeline), timeline)
    # reshape 2d -> final_dim
    if isinstance(model, FrozenLifetimeModel):
        ndim = max(map(np.ndim, (timeline, *model.args)), default=0)
    else:
        ndim = np.ndim(timeline)
    if ndim < 2:
        ls = np.squeeze(ls)
    return ls


class RenewalRewardProcess(RenewalProcess):

    def __init__(
        self,
        model: LifetimeModel[()],
        rewards: Rewards,
        discounting_rate: Optional[float] = None,
        *,
        model1: Optional[LifetimeModel[()]] = None,
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
