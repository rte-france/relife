from functools import partial, wraps
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from relife2.discount import Discount, ExponentialDiscounting
from relife2.model import LifetimeModel
from relife2.renewal.args import argscheck
from relife2.renewal.equation import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)
from relife2.renewal.sample import (
    GeneratedLifetime,
    GeneratedRewardLifetime,
    lifetimes_rewards_sampler,
    lifetimes_sampler,
)
from relife2.reward import Reward


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

    return model.ls_integrate(func, np.zeros_like(timeline), timeline, *model_args)


def exponentialdiscount(method: Callable) -> Callable:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if not isinstance(self.discount, ExponentialDiscounting):
            raise ValueError(
                "The discount must be of type ExponentielDiscounting for asymptotic expected_total_reward"
            )
        return method(self, *args, **kwargs)

    return wrapper


class RenewalProcess:

    generated_data = GeneratedLifetime

    def __init__(
        self,
        model: LifetimeModel,
        *,
        delayed_model: Optional[LifetimeModel] = None,
        nb_assets: Optional[int] = 1,
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
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
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
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
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

    @argscheck
    def sample(
        self,
        nb_samples: int,
        end_time: float,
        *,
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
    ) -> GeneratedLifetime:

        container = self.generated_data()
        for failure_times, lifetimes, still_valid in lifetimes_sampler(
            self.model,
            nb_samples,
            self.nb_assets,
            end_time=end_time,
            model_args=model_args,
            delayed_model=self.delayed_model,
            delayed_model_args=delayed_model_args,
        ):
            assets, samples = np.where(still_valid)
            container.update(
                failure_times[still_valid],
                lifetimes[still_valid],
                samples,
                assets,
            )
        container.close()
        return container


class RenewalRewardProcess(RenewalProcess):

    generated_data = GeneratedRewardLifetime
    discount = ExponentialDiscounting()

    def __init__(
        self,
        model: LifetimeModel,
        reward: Reward,
        *,
        delayed_model: Optional[LifetimeModel] = None,
        delayed_reward: Optional[Reward] = None,
        nb_assets: Optional[int] = 1,
    ):
        super().__init__(model, delayed_model=delayed_model, nb_assets=nb_assets)
        self.reward = reward
        self.delayed_reward = delayed_reward

        if self.delayed_model is None:
            self._reward_partial_expectation = partial(
                reward_partial_expectation,
                model=self.model,
                reward=self.reward,
                discount=self.discount,
            )
        else:
            if self.delayed_reward is None:
                self.delayed_reward = self.reward

            self._reward_partial_expectation = partial(
                reward_partial_expectation,
                model=self.delayed_model,
                reward=self.delayed_reward,
                discount=self.discount,
            )

    def asymptotic_expected_total_reward(
        self,
        *,
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        discount_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
    ):
        rate = discount_args[0]
        mask = rate <= 0
        rate = np.ma.MaskedArray(rate, mask)

        def f(x):
            return self.discount.factor(x, rate)

        def y(x):
            return self.discount.factor(x, rate) * self.reward(x, *reward_args)

        lf = self.model.ls_integrate(f, np.array(0.0), np.array(np.inf), *model_args)
        ly = self.model.ls_integrate(y, np.array(0.0), np.array(np.inf), *model_args)
        z = ly / (1 - lf)

        if self.delayed_model is not None:

            def y1(x):
                return self.discount.factor(x, rate) * self.reward(
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
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        discount_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
    ):

        z = renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func=self._reward_partial_expectation(
                timeline,
                model_args=model_args,
                reward_args=reward_args,
                disount_args=discount_args,
            ),
            model_args=model_args,
            discount_factor=self.discount,
            discount_factor_args=discount_args,
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
                    disount_args=discount_args,
                ),
                delayed_model_args=delayed_model_args,
                discount_factor=self.discount,
                discount_factor_args=discount_args,
            )

    def asymptotic_expected_equivalent_annual_worth(
        self,
        *,
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        discount_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
    ):
        rate = discount_args[0]
        mask = rate <= 0
        rate = np.ma.MaskedArray(rate, mask)

        q = rate * self.asymptotic_expected_total_reward(
            model_args=model_args,
            reward_args=reward_args,
            discount_args=discount_args,
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
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        discount_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
    ):

        z = self.expected_total_reward(
            timeline,
            model_args=model_args,
            reward_args=reward_args,
            discount_args=discount_args,
            delayed_model_args=delayed_model_args,
            delayed_reward_args=delayed_reward_args,
        )
        af = self.discount.annuity_factor(timeline, *discount_args)
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

    @argscheck
    def sample(
        self,
        nb_samples: int,
        end_time: float,
        *,
        model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        discount_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_model_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
        delayed_reward_args: Optional[
            NDArray[np.float64] | tuple[NDArray[np.float64], ...]
        ] = None,
    ):
        container = self.generated_data()
        for (
            failure_times,
            lifetimes,
            total_rewards,
            still_valid,
        ) in lifetimes_rewards_sampler(
            self.model,
            self.reward,
            self.discount,
            nb_samples,
            self.nb_assets,
            end_time=end_time,
            model_args=model_args,
            reward_args=reward_args,
            discount_args=discount_args,
            delayed_model=self.delayed_model,
            delayed_model_args=delayed_model_args,
            delayed_reward=self.delayed_reward,
            delayed_reward_args=delayed_reward_args,
        ):
            assets, samples = np.where(still_valid)
            container.update(
                failure_times[still_valid],
                lifetimes[still_valid],
                total_rewards[still_valid],
                samples,
                assets,
            )
        container.close()
        return container
