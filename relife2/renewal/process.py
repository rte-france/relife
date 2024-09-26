from functools import partial
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife2.core import LifetimeModel
from relife2.renewal.args import ArgsDict, argscheck
from relife2.renewal.discounts import Discount, ExponentialDiscounting
from relife2.renewal.equations import renewal_equation_solver
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

    return model.ls_integrate(func, 0, timeline, *model_args)


class RenewalLifetimeProcess:

    def __init__(
        self,
        model: LifetimeModel,
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
        self, timeline: NDArray[np.float64], args: Optional[ArgsDict] = None
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
        self, timeline: NDArray[np.float64], args: Optional[ArgsDict] = None
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
        initmodel: Optional[LifetimeModel] = None,
        initreward: Optional[LifetimeModel] = None,
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

            self._expected_total_reward = partial(
                renewal_equation_solver,
                model=self.model,
                evaluated_func=self._reward_partial_expectation,
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

            self._expected_total_reward = partial(
                renewal_equation_solver,
                model=self.initmodel,
                evaluated_func=self._reward_partial_expectation,
            )

    def _asymptotic_expected_total_reward(
        self, timeline: NDArray[np.float64], args: ArgsDict
    ):
        pass

    @argscheck
    def expected_total_reward(
        self,
        timeline: NDArray[np.float64],
        args: Optional[ArgsDict] = None,
        asymptotic: bool = False,
    ):
        if args is None:
            args = {}
        if asymptotic:
            return self._asymptotic_expected_total_reward(timeline, args)

        if self.initmodel is None:
            return self._expected_total_reward(
                timeline,
                model_args=args.get("model", ()),
                evaluated_func_args=(
                    args.get("model"),
                    args.get("reward"),
                    args.get("discount"),
                ),
            )
        else:
            return self._expected_total_reward(
                timeline,
                model_args=args.get("initmodel", ()),
                evaluated_func_args=(
                    args.get("initmodel"),
                    args.get("initreward"),
                    args.get("discount"),
                ),
            )

    def _asymptotic_expected_equivalent_annual_worth(
        self, timeline: NDArray[np.float64], args: ArgsDict
    ):
        pass

    @argscheck
    def expected_equivalent_annual_worth(
        self,
        timeline: NDArray[np.float64],
        args: Optional[ArgsDict] = None,
        asymptotic: bool = False,
    ):
        if args is None:
            args = {}
        if asymptotic:
            return self._asymptotic_expected_equivalent_annual_worth(timeline, args)
        pass

    @argscheck
    def asymptotic_expected_equivalent_annual_worth(
        self, timeline: NDArray[np.float64], args: Optional[ArgsDict] = None
    ):
        pass

    @argscheck
    def sample(self, nb_samples, end_time, args: Optional[ArgsDict] = None):
        if args is None:
            args = {}
        pass
