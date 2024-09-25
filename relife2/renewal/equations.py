from functools import partial
from typing import Callable, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife2.core import LifetimeModel
from relife2.renewal.discounts import Discount, ExponentialDiscounting
from relife2.renewal.rewards import Reward
from relife2.renewal.sample import LifetimesSampler

Ts = TypeVarTuple("Ts")


def renewal_equation_solver(
    timeline: NDArray[np.float64],
    model: LifetimeModel,
    evaluated_func: Optional[Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]]],
    model_args: tuple[*Ts] = (),
    evaluated_func_args: tuple[*Ts] = (),
    discount_factor: Optional[
        Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]]
    ] = None,
    discount_factor_args: tuple[*Ts] = (),
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f = model.cdf(timeline, *model_args)
    fm = model.cdf(tm, *model_args)
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


def delayed_renewal(
    timeline: NDArray[np.float64],
    z: NDArray[np.float64],
    model1: LifetimeModel,
    evaluated_func: Optional[Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]]],
    model1_args: tuple[*Ts] = (),
    evaluated_func_args: tuple[*Ts] = (),
    discount_factor: Optional[
        Callable[[NDArray[np.float64], *Ts], NDArray[np.float64]]
    ] = None,
    discount_factor_args: tuple[*Ts] = (),
) -> NDArray[np.float64]:

    tm = 0.5 * (timeline[1:] + timeline[:-1])
    f1 = model1.cdf(timeline, *model1_args)
    f1m = model1.cdf(tm, *model1_args)
    y1 = evaluated_func(timeline, *evaluated_func_args)

    if discount_factor is not None:
        d = discount_factor(timeline, *discount_factor_args)
    else:
        d = np.ones_like(f1)
    z1 = np.empty(y1.shape)
    u1 = d * np.insert(f1[:, 1:] - f1m, 0, 1, axis=-1)
    v1 = d[:, :-1] * np.insert(np.diff(f1m), 0, 1, axis=-1)
    z1[:, 0] = y1[:, 0]
    z1[:, 1] = y1[:, 1] + z[:, 0] * u1[:, 1] + z[:, 1] * d[:, 0] * f1m[:, 0]
    for n in range(2, f1.shape[-1]):
        z1[:, n] = (
            y1[:, n]
            + z[:, 0] * u1[:, n]
            + z[:, n] * d[:, 0] * f1m[:, 0]
            + np.sum(z[:, 1:n][:, ::-1] * v1[:, 1:n], axis=-1)
        )
    return z1


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel,
    reward: Reward,
    discount: Discount,
    model_args: tuple[NDArray[np.float64], ...] = (),
    reward_args: tuple[NDArray[np.float64], ...] = (),
    discount_args: tuple[NDArray[np.float64], ...] = (),
) -> np.ndarray:

    def func(x):
        return reward(x, *reward_args) * discount.factor(x, *discount_args)

    return model.ls_integrate(func, 0, timeline, *model_args)


# TODO : as in sample, model_args, initmodel_args must be passed as method parameters
# maybe pass one args : dict[str, tuple[NDArray[np.float64], ...]] to reduce number of parameters
# then use args.get("model_args", ()) for the rest of the code
# TODO : then remove add_initmodel (initmodel is just an optional param in constructor)
# TODO : add nb_assets as attrib to follow sampler logic and fix expected nb assets (checkargs depends on it)
# argue that initmodel may be distri and model regression => must know in advance nb assets to infer rvs_size
# TODO : add checkargs decorator


class RenewalLifetimeProcess:

    def __init__(
        self, model: LifetimeModel, model_args: tuple[NDArray[np.float64], ...] = ()
    ):
        self.model = model
        self.model_args = model_args
        self.initmodel = None
        self.initmodel_args = None

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

    def add_initmodel(
        self, model: LifetimeModel, model_args: tuple[NDArray[np.float64], ...] = ()
    ):
        self.initmodel = model
        self.initmodel_args = model_args

        self._renewal_function = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self.initmodel.cdf,
            cdf_args=self.model_args,
            evaluated_func_args=self.initmodel_args,
        )

        self._renewal_density = partial(
            renewal_equation_solver,
            cdf=self.model.cdf,
            evaluated_func=self.initmodel.pdf,
            cdf_args=self.model_args,
            evaluated_func_args=self.initmodel_args,
        )

    def renewal_function(self, timeline: NDArray[np.float64]):
        return self._renewal_function(timeline)

    def renewal_density(self, timeline: NDArray[np.float64]):
        return self._renewal_density(timeline)

    def sample(
        self, nb_samples: int, end_time: float, nb_assets: int = 1
    ) -> LifetimesSampler:
        sampler = LifetimesSampler(
            self.model, nb_assets=nb_assets, initmodel=self.initmodel
        )
        sampler.sample(
            nb_samples,
            end_time,
            model_args=self.model_args,
            initmodel_args=self.initmodel_args,
        )
        return sampler


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

        self._reward_partial_expectation = partial(
            reward_partial_expectation,
            model=self.model,
            reward=self.reward,
            discount=self.discount,
            model_args=self.model_args,
            reward_args=self.reward_args,
            discount_args=self.discount_args,
        )

        self._expected_total_reward = partial(
            renewal_equation_solver,
            model=self.model,
            evaluated_func=self._reward_partial_expectation,
            model_args=self.model_args,
            evaluated_func_args=(
                self.model_args,
                self.reward_args,
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

        self._reward_partial_expectation = partial(
            reward_partial_expectation,
            model=self.model1,
            reward=self.reward1,
            discount=self.discount,
            model_args=self.model1_args,
            reward_args=self.reward1_args,
            discount_args=self.discount_args,
        )

        self._expected_total_reward = partial(
            renewal_equation_solver,
            model=self.model1,
            evaluated_func=self._reward_partial_expectation,
            model_args=self.model1_args,
            evaluated_func_args=(
                self.model1_args,
                self.reward1_args,
                self.discount_args,
            ),
        )

    def expected_total_reward(self, timeline: NDArray[np.float64]):
        return self._expected_total_reward(timeline)
