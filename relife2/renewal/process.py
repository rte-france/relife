from functools import partial, wraps
from typing import Optional, Callable

import numpy as np
from numpy.typing import NDArray

from relife2.core import LifetimeModel
from relife2.renewal.args import ArgsDict, argscheck
from relife2.renewal.discounts import Discount, ExponentialDiscounting
from relife2.renewal.equations import (
    renewal_equation_solver,
    delayed_renewal_equation_solver,
)
from relife2.renewal.rewards import Reward
from relife2.renewal.sample import LifetimesSampler


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


class RenewalLifetimeProcess:

    def __init__(
        self,
        model: LifetimeModel,
        *,
        initmodel: Optional[LifetimeModel] = None,
        nb_assets: Optional[int] = 1,
    ):
        self.model = model
        self.initmodel = initmodel
        self.nb_assets = nb_assets

        if self.initmodel is None:
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
                evaluated_func=self.initmodel.cdf,
            )

            self._renewal_density = partial(
                renewal_equation_solver,
                cdf=self.model.cdf,
                evaluated_func=self.initmodel.pdf,
            )

    @argscheck
    def renewal_function(
        self, timeline: NDArray[np.float64], *, args: Optional[ArgsDict] = None
    ):
        if args is None:
            args = {}
        if self.initmodel is None:
            evaluated_func_args = args.get("model", ())
        else:
            evaluated_func_args = args.get("initmodel", ())
        return self._renewal_function(
            timeline,
            cdf_args=args.get("model", ()),
            evaluated_func_args=evaluated_func_args,
        )

    @argscheck
    def renewal_density(
        self, timeline: NDArray[np.float64], *, args: Optional[ArgsDict] = None
    ):
        if args is None:
            args = {}
        if self.initmodel is None:
            evaluated_func_args = args.get("model", ())
        else:
            evaluated_func_args = args.get("initmodel", ())
        return self._renewal_density(
            timeline,
            cdf_args=args.get("model", ()),
            evaluated_func_args=evaluated_func_args,
        )

    @argscheck
    def sample(
        self,
        nb_samples: int,
        end_time: float,
        *,
        args: Optional[ArgsDict] = None,
    ) -> LifetimesSampler:
        if args is None:
            args = {}
        sampler = LifetimesSampler(
            self.model, nb_assets=self.nb_assets, initmodel=self.initmodel
        )
        sampler.sample(
            nb_samples,
            end_time,
            model_args=args.get("model", ()),
            initmodel_args=args.get("initmodel", ()),
        )
        return sampler


class RenewalRewardLifetimeProcess:
    def __init__(
        self,
        model: LifetimeModel,
        reward: Reward,
        discount: Discount = ExponentialDiscounting(),
        *,
        initmodel: Optional[LifetimeModel] = None,
        initreward: Optional[Reward] = None,
        nb_assets: Optional[int] = 1,
    ):
        self.model = model
        self.reward = reward
        self.discount = discount
        self.initmodel = initmodel
        self.initreward = initreward
        self.nb_assets = nb_assets

        if self.initmodel is None:
            self._reward_partial_expectation = partial(
                reward_partial_expectation,
                model=self.model,
                reward=self.reward,
                discount=self.discount,
            )
        else:
            if self.initreward is None:
                self.initreward = self.reward

            self._reward_partial_expectation = partial(
                reward_partial_expectation,
                model=self.initmodel,
                reward=self.initreward,
                discount=self.discount,
            )

    @exponentialdiscount
    def _asymptotic_expected_total_reward(self, args: ArgsDict):
        rate = args.get("discount", (0.0,))[0]
        mask = rate <= 0
        rate = np.ma.MaskedArray(rate, mask)

        def f(x):
            return self.discount.factor(x, rate)

        def y(x):
            return self.discount.factor(x, rate) * self.reward(
                x, *args.get("reward", ())
            )

        lf = self.model.ls_integrate(
            f, np.array(0.0), np.array(np.inf), *args.get("model", ())
        )
        ly = self.model.ls_integrate(
            y, np.array(0.0), np.array(np.inf), *args.get("model", ())
        )
        z = ly / (1 - lf)

        if self.initmodel is not None:

            def y1(x):
                return self.discount.factor(x, rate) * self.reward(
                    x, *args.get("initreward", ())
                )

            lf1 = self.initmodel.ls_integrate(
                f, np.array(0.0), np.array(np.inf), *args.get("initmodel", ())
            )
            ly1 = self.initmodel.ls_integrate(
                y1, np.array(0.0), np.array(np.inf), *args.get("initmodel", ())
            )
            z = ly1 + z * lf1
        return np.where(mask, np.inf, z)

    @argscheck
    def expected_total_reward(
        self,
        timeline: Optional[NDArray[np.float64]] = None,
        /,
        *,
        args: Optional[ArgsDict] = None,
        asymptotic: bool = False,
    ):
        if args is None:
            args = {}
        if asymptotic:
            return self._asymptotic_expected_total_reward(args)
        else:
            if timeline is None:
                raise ValueError(
                    "If not asymptotic expected_total_reward requires timeline as first argument"
                )

        z = renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func=self._reward_partial_expectation(
                timeline,
                model_args=args.get("model", ()),
                reward_args=args.get("reward", ()),
                disount_args=args.get("discount", ()),
            ),
            model_args=args.get("model", ()),
            discount_factor=self.discount,
            discount_factor_args=args.get("discount", ()),
        )

        if self.initmodel is None:
            return z
        else:
            return delayed_renewal_equation_solver(
                timeline,
                z,
                self.initmodel,
                evaluated_func=self._reward_partial_expectation(
                    timeline,
                    model_args=args.get("initmodel", ()),
                    reward_args=args.get("initreward", ()),
                    disount_args=args.get("discount", ()),
                ),
                initmodel_args=args.get("initmodel", ()),
                discount_factor=self.discount,
                discount_factor_args=args.get("discount", ()),
            )

    @exponentialdiscount
    def _asymptotic_expected_equivalent_annual_worth(self, args: ArgsDict):
        rate = args.get("discount", (0.0,))[0]
        mask = rate <= 0
        rate = np.ma.MaskedArray(rate, mask)

        q = rate * self._asymptotic_expected_total_reward(args)
        q0 = self.model.ls_integrate(
            lambda x: self.reward(x, *args.get("reward", ())),
            np.array(0.0),
            np.array(np.inf),
            *args.get("model", ()),
        ) / self.model.mean(*args.get("model", ()))
        return np.where(mask, q0, q)

    @argscheck
    def expected_equivalent_annual_worth(
        self,
        timeline: Optional[NDArray[np.float64]] = None,
        /,
        *,
        args: Optional[ArgsDict] = None,
        asymptotic: bool = False,
    ):
        if args is None:
            args = {}
        if asymptotic:
            return self._asymptotic_expected_equivalent_annual_worth(args)
        else:
            if timeline is None:
                raise ValueError(
                    "If not asymptotic expected_equivalent_annual_worth requires timeline as first argument"
                )

        z = self.expected_total_reward(timeline, args=args, asymptotic=False)
        af = self.discount.annuity_factor(timeline, *args.get("discount", ()))
        mask = af == 0.0
        af = np.ma.masked_where(mask, af)
        q = z / af
        if self.initmodel is None:
            q0 = self.reward(np.array(0.0), *args.get("reward", ())) * self.model.pdf(
                np.array(0.0), *args.get("model", ())
            )
        else:
            q0 = self.initreward(
                np.array(0.0), *args.get("initreward", ())
            ) * self.initmodel.pdf(np.array(0.0), *args.get("initmodel", ()))
        return np.where(mask, q0, q)

    @argscheck
    def sample(self, nb_samples, end_time, args: Optional[ArgsDict] = None):
        if args is None:
            args = {}
        rewards = (
            np.array(
                self.reward(
                    data.durations.reshape(-1, 1),
                    *args_take(data.indices, *reward_args),
                ).swapaxes(-2, -1)
                * self.discount.factor(data.times, *discount_args),
                ndmin=3,
            )
            .sum(axis=0)
            .ravel()
        )
