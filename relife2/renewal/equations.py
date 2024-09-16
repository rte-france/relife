from functools import partial
from typing import Callable, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife2.core import LifetimeModel
from relife2.renewal.discounts import Discount, ExponentialDiscounting
from relife2.renewal.rewards import Reward

Ts = TypeVarTuple("Ts")


class RenewalLifetimeProcess:

    def __init__(
        self, model: LifetimeModel, model_args: tuple[NDArray[np.float64], ...] = ()
    ):
        self.model = model
        self.model_args = model_args
        self.model1 = None
        self.model1_args = None

        self._renewal_function = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self.model.cdf,
            cdf_args=self.model_args,
            evaluated_func_args=self.model_args,
        )

        self._renewal_density = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self.model.pdf,
            cdf_args=self.model_args,
            evaluated_func_args=self.model_args,
        )

    def add_init_model(
        self, model: LifetimeModel, model_args: tuple[NDArray[np.float64], ...] = ()
    ):
        self.model1 = model
        self.model1_args = model_args

        self._renewal_function = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self.model1.cdf,
            cdf_args=self.model_args,
            evaluated_func_args=self.model1_args,
        )

        self._renewal_density = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self.model1.pdf,
            cdf_args=self.model_args,
            evaluated_func_args=self.model1_args,
        )

    def renewal_function(self, timeline: NDArray[np.float64]):
        return self._renewal_function(timeline)

    def renewal_density(self, timeline: NDArray[np.float64]):
        return self._renewal_density(timeline)


class RenewalRewardLifetimeProcess:
    def __init__(
        self,
        model: LifetimeModel,
        reward: Reward,
        discount: Discount = ExponentialDiscounting(),
        model_args: tuple[NDArray[np.float64], ...] = (),
        reward_args: tuple[NDArray[np.float64], ...] = (),
        discount_args: tuple[NDArray[np.float64], ...] = (),
    ):
        self.model = model
        self.reward = reward
        self.discount = discount
        self.model_args = model_args
        self.reward_args = reward_args
        self.discount_args = discount_args

        self.model1 = None
        self.reward1 = None
        self.model1_args = None
        self.reward1_args = None

        self._expected_total_reward = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self._reward_partial_expectation,
            cdf_args=self.model_args,
            evaluated_func_args=(
                self.model_args,
                self.reward1_args,
                self.discount_args,
            ),
        )

    def add_init_model(
        self,
        model: LifetimeModel,
        reward: Reward,
        model_args: tuple[NDArray[np.float64], ...] = (),
        reward_args: tuple[NDArray[np.float64], ...] = (),
    ):
        self.model1 = model
        self.model1_args = model_args
        self.reward1 = reward
        self.reward1_args = reward_args

    @classmethod
    def _reward_partial_expectation(
        cls,
        t: np.ndarray,
        model: LifetimeModel,
        reward: Reward,
        discount: Discount,
        model_args: tuple[NDArray[np.float64], ...] = (),
        reward_args: tuple[NDArray[np.float64], ...] = (),
        discount_args: tuple[NDArray[np.float64], ...] = (),
    ) -> np.ndarray:

        func = lambda x: reward(x, *reward_args) * discount.factor(x, *discount_args)
        ndim = args_ndim(t, *model_args, *reward_args, *discount_args)
        return ls_integrate(func, 0, t, *model_args, ndim=ndim)


def renewal_equation_solver(
    timeline: NDArray[np.float64],
    cdf: Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]],
    evaluated_func: Optional[Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]]],
    cdf_args: tuple[*Ts] = (),
    evaluated_func_args: tuple[*Ts] = (),
    discount_factor: Optional[
        Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]]
    ] = None,
    discount_factor_args: tuple[*Ts] = (),
):

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f = cdf(timeline, *cdf_args)
    fm = cdf(tm, *cdf_args)
    y = evaluated_func(timeline, *evaluated_func_args)
    if discount_factor is not None:
        d = discount_factor(timeline, *discount_factor_args)
    else:
        d = np.ones_like(f)
    z = np.empty(y.shape)
    u = d * np.insert(f[:, 1:] - fm, 0, 1, axis=-1)
    v = d[:, :-1] * np.insert(np.diff(fm), 0, 1, axis=-1)
    q0 = 1 / (1 - d[:, 0] * fm[:, 0])
    z[:, 0] = y[:, 0]
    z[:, 1] = q0 * (y[:, 1] + z[:, 0] * u[:, 1])
    for n in range(2, f.shape[-1]):
        z[:, n] = q0 * (
            y[:, n]
            + z[:, 0] * u[:, n]
            + np.sum(z[:, 1:n][:, ::-1] * v[:, 1:n], axis=-1)
        )
    return z
