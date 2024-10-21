from functools import partial, wraps
from typing import Callable, Generic, Optional, ParamSpec, TypedDict, TypeVar

import numpy as np
from numpy.typing import NDArray

from relife2.model import LifetimeModel
from relife2.renewal.discounts import Discount, exponential_discount
from relife2.renewal.equations import (
    delayed_renewal_equation_solver,
    renewal_equation_solver,
)
from relife2.renewal.rewards import Reward

M = TypeVar("M", tuple[NDArray[np.float64], ...], tuple[()])
M1 = TypeVar("M1", tuple[NDArray[np.float64], ...], tuple[()])
R = TypeVar("R", tuple[NDArray[np.float64], ...], tuple[()])
R1 = TypeVar("R1", tuple[NDArray[np.float64], ...], tuple[()])
D = TypeVar("D", tuple[NDArray[np.float64], ...], tuple[()])

P = ParamSpec("P")
T = TypeVar("T")


def argscheck(method: Callable[P, T]) -> Callable[P, T]:
    @wraps(method)
    def wrapper(self, *args: P.args, **kwargs: P.kwargs) -> T:
        for key, value in kwargs.items():
            if "args" in key and value is not None:
                if not isinstance(value, tuple):
                    raise ValueError(
                        f"args must parameters must be filled in tuple : got {type(value)} for {key}"
                    )
                if self.nb_assets > 1:
                    for array in value:
                        if not isinstance(array, np.ndarray):
                            raise ValueError(
                                f"args values must be ndarray only : got {type(array)} for {key}"
                            )
                        if array.ndim != 2:
                            raise ValueError(
                                f"If nb_assets is {self.nb_assets}, args array must have 2 dimensions"
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


class RenewalProcessArgs(TypedDict, Generic[M, M1]):
    model_args: M
    delayed_model_args: M1


# make partial in method only : do not split in multiple functions
class RenewalProcess(Generic[M, M1]):

    args: RenewalProcessArgs

    def __init__(
        self,
        model: LifetimeModel[*M],
        *,
        nb_assets: int = 1,
        model_args: M = (),
        model1: Optional[LifetimeModel[*M1]] = None,
        delayed_model_args: M1 = (),
    ):
        self.model = model
        self.model1 = model1
        self.nb_assets = nb_assets
        self.args["model_args"] = model_args
        self.args["delayed_model_args"] = delayed_model_args

    def renewal_function(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model1 is None:
            evaluated_func = partial(
                lambda time, args: self.model.cdf(time, *args),
                args=self.args["model_args"],
            )
        else:
            evaluated_func = partial(
                lambda time, args: self.model1.cdf(time, *args),
                args=self.args["delayed_model_args"],
            )
        return renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func,
            model_args=self.args["model_args"],
        )

    def renewal_density(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.model1 is None:
            evaluated_func = partial(
                lambda time, args: self.model.pdf(time, *args),
                args=self.args["model_args"],
            )
        else:
            evaluated_func = partial(
                lambda time, args: self.model1.pdf(time, *args),
                args=self.args["delayed_model_args"],
            )
        return renewal_equation_solver(
            timeline,
            self.model,
            evaluated_func,
            model_args=self.args["model_args"],
        )


def reward_partial_expectation(
    timeline: NDArray[np.float64],
    model: LifetimeModel[*M],
    reward: Reward[*R],
    discount: Discount[*D],
    *,
    model_args: M = (),
    reward_args: R = (),
    discount_args: D = (),
) -> np.ndarray:

    def func(x):
        return reward(x, *reward_args) * discount.factor(x, *discount_args)

    return model.ls_integrate(func, np.zeros_like(timeline), timeline, *model_args)


class RenewalRewardProcessArgs(TypedDict, Generic[M, M1, R, R1]):
    model_args: M
    delayed_model_args: M1
    discount_rate: float | NDArray[np.float64]
    reward_args: R
    delayed_reward_args: R1


class RenewalRewardProcess(RenewalProcess[M, M1], Generic[M, M1, R, R1]):

    args: RenewalRewardProcessArgs

    def __init__(
        self,
        model: LifetimeModel[*M],
        reward: Reward[*R],
        *,
        nb_assets: int = 1,
        model_args: M = (),
        reward_args: R = (),
        discount_rate: float | NDArray[np.float64] = 0.0,
        model1: Optional[LifetimeModel[*M1]] = None,
        delayed_model_args: M1 = (),
        reward1: Optional[Reward[*R1]] = None,
        delayed_reward_args: R1 = (),
    ):
        super().__init__(model, nb_assets=nb_assets, model1=model1)
        self.reward = reward
        self.reward1 = reward1
        self.args["model_args"] = model_args
        self.args["delayed_model_args"] = delayed_model_args
        self.args["reward_args"] = reward_args
        self.args["discount_rate"] = discount_rate
        self.args["delayed_reward_args"] = delayed_reward_args

    def expected_total_reward(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        z = renewal_equation_solver(
            timeline,
            self.model,
            partial(
                reward_partial_expectation,
                model=self.model,
                reward=self.reward,
                discount=exponential_discount,
                model_args=self.args["model_args"],
                reward_args=self.args["reward_args"],
                disounting_args=(self.args["discount_rate"],),
            ),
            model_args=self.args["model_args"],
            discount=exponential_discount,
            discount_args=(self.args["discount_rate"],),
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
                    reward=self.reward1,
                    discount=exponential_discount,
                    model_args=self.args["delayed_model_args"],
                    reward_args=self.args["delayed_reward_args"],
                    disount_args=(self.args["discount_rate"],),
                ),
                delayed_model_args=self.args["delayed_model_args"],
                discount=exponential_discount,
                discount_args=(self.args["discount_rate"],),
            )

    def asymptotic_expected_total_reward(
        self,
    ) -> NDArray[np.float64]:
        mask = self.args["discount_rate"] <= 0
        rate = np.ma.MaskedArray(self.args["discount_rate"], mask)

        def f(x):
            return exponential_discount.factor(x, rate)

        def y(x):
            return exponential_discount.factor(x, rate) * self.reward(
                x, *self.args["reward_args"]
            )

        lf = self.model.ls_integrate(
            f, np.array(0.0), np.array(np.inf), *self.args["model_args"]
        )
        ly = self.model.ls_integrate(
            y, np.array(0.0), np.array(np.inf), *self.args["model_args"]
        )
        z = ly / (1 - lf)

        if self.model1 is not None:

            def y1(x):
                return exponential_discount.factor(x, rate) * self.reward(
                    x, *self.args["delayed_reward_args"]
                )

            lf1 = self.model1.ls_integrate(
                f, np.array(0.0), np.array(np.inf), *self.args["delayed_model_args"]
            )
            ly1 = self.model1.ls_integrate(
                y1, np.array(0.0), np.array(np.inf), *self.args["delayed_model_args"]
            )
            z = ly1 + z * lf1
        return np.where(mask, np.inf, z)

    def expected_equivalent_annual_worth(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:

        z = self.expected_total_reward(timeline)
        af = exponential_discount.annuity_factor(timeline, self.args["discount_rate"])
        mask = af == 0.0
        af = np.ma.masked_where(mask, af)
        q = z / af
        if self.model1 is None:
            q0 = self.reward(np.array(0.0), *self.args["reward_args"]) * self.model.pdf(
                np.array(0.0), *self.args["model_args"]
            )
        else:
            q0 = self.reward1(
                np.array(0.0), *self.args["delayed_reward_args"]
            ) * self.model1.pdf(np.array(0.0), *self.args["delayed_model_args"])
        return np.where(mask, q0, q)

    def asymptotic_expected_equivalent_annual_worth(self) -> NDArray[np.float64]:
        mask = self.args["discount_rate"] <= 0
        discount_rate = np.ma.MaskedArray(self.args["discount_rate"], mask)

        q = discount_rate * self.asymptotic_expected_total_reward()
        q0 = self.model.ls_integrate(
            lambda x: self.reward(x, *self.args["reward_args"]),
            np.array(0.0),
            np.array(np.inf),
            *self.args["model_args"],
        ) / self.model.mean(*self.args["model_args"])
        return np.where(mask, q0, q)
