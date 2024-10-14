from functools import partial, wraps
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discounting import exponential_discounting, Discounting
from relife2.renewal.equation import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)
from relife2.renewal.reward import Reward


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel,
    reward: Reward,
    discounting: Discounting,
    model_args: tuple[NDArray[np.float64], ...] = (),
    reward_args: tuple[NDArray[np.float64], ...] = (),
    discounting_args: tuple[NDArray[np.float64], ...] = (),
) -> np.ndarray:

    def func(x):
        return reward(x, *reward_args) * discounting.factor(x, *discounting_args)

    return model.ls_integrate(func, np.zeros_like(timeline), timeline, *model_args)


def argscheck(
    method: Callable[..., NDArray[np.float64]]
) -> Callable[..., NDArray[np.float64]]:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        for key, value in kwargs.items():
            if "args" in key and value is not None:
                if isinstance(value, np.ndarray):
                    value = (value,)
                if self.nb_assets > 1:
                    for array in value:
                        if array.ndim != 2:
                            raise ValueError(
                                f"If nb_assets is more than 1 (got {self.nb_assets}), args array must have 2 dimensions"
                            )
                        if array.shape[0] != self.nb_assets:
                            raise ValueError(
                                f"Expected {self.nb_assets} nb assets but got {array.shape[0]} in {key}"
                            )
                else:
                    for array in value:
                        if array.ndim > 1:
                            raise ValueError(
                                f"If nb_assets is 1, args array cannot have more than 1 dimension"
                            )
        return method(self, *args, **kwargs)

    return wrapper


class RenewalProcess:

    def __init__(
        self,
        model: LifetimeModel,
        *,
        nb_assets: int = 1,
        delayed_model: Optional[LifetimeModel] = None,
    ):
        self.model = model
        self.delayed_model = delayed_model
        self.nb_assets = nb_assets

        if self.delayed_model is None:
            self._renewal_function = partial(
                renewal_equation_solver,
                cdf=self.model.cdf,
                evaluated_func=self.model.cdf,
            )

            self._renewal_density = partial(
                renewal_equation_solver,
                cdf=self.model.cdf,
                evaluated_func=self.model.pdf,
            )
        else:
            self._renewal_function = partial(
                renewal_equation_solver,
                cdf=self.model.cdf,
                evaluated_func=self.delayed_model.cdf,
            )

            self._renewal_density = partial(
                renewal_equation_solver,
                cdf=self.model.cdf,
                evaluated_func=self.delayed_model.pdf,
            )

    @argscheck
    def renewal_function(
        self,
        timeline: NDArray[np.float64],
        *,
        model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
    ):
        if self.delayed_model is None:
            evaluated_func_args = model_args
        else:
            evaluated_func_args = delayed_model_args
        return self._renewal_function(
            timeline,
            cdf_args=model_args,
            evaluated_func_args=evaluated_func_args,
        )

    @argscheck
    def renewal_density(
        self,
        timeline: NDArray[np.float64],
        *,
        model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
    ):
        if self.delayed_model is None:
            evaluated_func_args = model_args
        else:
            evaluated_func_args = delayed_model_args
        return self._renewal_density(
            timeline,
            cdf_args=model_args,
            evaluated_func_args=evaluated_func_args,
        )


class RenewalRewardProcess(RenewalProcess):

    def __init__(
        self,
        model: LifetimeModel,
        reward: Reward,
        *,
        nb_assets: int = 1,
        delayed_model: Optional[LifetimeModel] = None,
        delayed_reward: Optional[Reward] = None,
    ):
        super().__init__(model, nb_assets=nb_assets, delayed_model=delayed_model)
        self.reward = reward
        self.delayed_reward = delayed_reward

        if self.delayed_model is None:
            self._reward_partial_expectation = partial(
                reward_partial_expectation,
                model=self.model,
                reward=self.reward,
                discounting=exponential_discounting,
            )
        else:
            if self.delayed_reward is None:
                self.delayed_reward = self.reward

            self._reward_partial_expectation = partial(
                reward_partial_expectation,
                model=self.delayed_model,
                reward=self.delayed_reward,
                discounting=exponential_discounting,
            )

    @argscheck
    def asymptotic_expected_total_reward(
        self,
        *,
        model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        discount_rate: float | NDArray[np.float64] = 0.0,
        delayed_model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
    ):
        mask = discount_rate <= 0
        rate = np.ma.MaskedArray(discount_rate, mask)

        def f(x):
            return exponential_discounting.factor(x, rate)

        def y(x):
            return exponential_discounting.factor(x, rate) * self.reward(
                x, *reward_args
            )

        lf = self.model.ls_integrate(f, np.array(0.0), np.array(np.inf), *model_args)
        ly = self.model.ls_integrate(y, np.array(0.0), np.array(np.inf), *model_args)
        z = ly / (1 - lf)

        if self.delayed_model is not None:

            def y1(x):
                return exponential_discounting.factor(x, rate) * self.reward(
                    x, *delayed_reward_args
                )

            lf1 = self.delayed_model.ls_integrate(
                f, np.array(0.0), np.array(np.inf), *delayed_model_args
            )
            ly1 = self.delayed_model.ls_integrate(
                y1, np.array(0.0), np.array(np.inf), *delayed_model_args
            )
            z = ly1 + z * lf1
        return np.where(mask, np.inf, z)

    @argscheck
    def expected_total_reward(
        self,
        timeline: NDArray[np.float64],
        /,
        *,
        discount_rate: float | NDArray[np.float64] = 0.0,
        model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
    ):

        z = renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func=self._reward_partial_expectation(
                timeline,
                model_args=model_args,
                reward_args=reward_args,
                disounting_args=(discount_rate,),
            ),
            model_args=model_args,
            discounting=exponential_discounting,
            discounting_args=(discount_rate,),
        )

        if self.delayed_model is None:
            return z
        else:
            return delayed_renewal_equation_solver(
                timeline,
                z,
                self.delayed_model,
                evaluated_func=self._reward_partial_expectation(
                    timeline,
                    model_args=delayed_model_args,
                    reward_args=delayed_reward_args,
                    disount_args=(discount_rate,),
                ),
                delayed_model_args=delayed_model_args,
                discounting=exponential_discounting,
                discounting_args=(discount_rate,),
            )

    @argscheck
    def asymptotic_expected_equivalent_annual_worth(
        self,
        *,
        model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        discount_rate: float | NDArray[np.float64] = 0.0,
        delayed_model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
    ):
        mask = discount_rate <= 0
        discount_rate = np.ma.MaskedArray(discount_rate, mask)

        q = discount_rate * self.asymptotic_expected_total_reward(
            model_args=model_args,
            reward_args=reward_args,
            discounting_args=(discount_rate,),
            delayed_model_args=delayed_model_args,
            delayed_reward_args=delayed_reward_args,
        )
        q0 = self.model.ls_integrate(
            lambda x: self.reward(x, *reward_args),
            np.array(0.0),
            np.array(np.inf),
            *model_args,
        ) / self.model.mean(*model_args)
        return np.where(mask, q0, q)

    @argscheck
    def expected_equivalent_annual_worth(
        self,
        timeline: NDArray[np.float64],
        /,
        *,
        model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        discount_rate: float | NDArray[np.float64] = 0.0,
        delayed_model_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
        delayed_reward_args: NDArray[np.float64] | tuple[NDArray[np.float64], ...] = (),
    ):

        z = self.expected_total_reward(
            timeline,
            model_args=model_args,
            reward_args=reward_args,
            discounting_args=(discount_rate,),
            delayed_model_args=delayed_model_args,
            delayed_reward_args=delayed_reward_args,
        )
        af = exponential_discounting.annuity_factor(timeline, discount_rate)
        mask = af == 0.0
        af = np.ma.masked_where(mask, af)
        q = z / af
        if self.delayed_model is None:
            q0 = self.reward(np.array(0.0), *reward_args) * self.model.pdf(
                np.array(0.0), *model_args
            )
        else:
            q0 = self.delayed_reward(
                np.array(0.0), *delayed_reward_args
            ) * self.delayed_model.pdf(np.array(0.0), *delayed_model_args)
        return np.where(mask, q0, q)
