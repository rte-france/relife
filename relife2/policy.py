import functools
from typing import Optional, Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import newton

from relife2.fiability import AgeReplacementModel, LeftTruncatedModel, LifetimeModel
from relife2.renewal import (
    age_replacement_cost,
    exponential_discount,
    lifetimes_rewards_generator,
    run_to_failure_cost,
)
from relife2.utils.data import RenewalRewardData
from relife2.utils.types import Model1Args, ModelArgs
from .nhpp import NHPP
from .renewalprocess import RenewalRewardProcess, reward_partial_expectation
from .utils.integration import gauss_legendre


class Policy(Protocol):
    """
    structural typing of policy object
    """

    def expected_total_cost(
        self, timeline: NDArray[np.float64]  # tf: float, period:float=1
    ) -> NDArray[np.float64]:
        """warning: tf > 0, period > 0, dt is deduced from period and is < 0.5"""

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


def ifset(*param_names: str):
    """
    simple decorator to check if some params are set before executed one method
    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            for name in param_names:
                if getattr(self, name) is None:
                    raise ValueError(
                        f"{name} is not set. If fit exists, you may need to fit the policy first"
                    )
            return method(self, *args, **kwargs)

        return wrapper

    return decorator


class OneCycleRunToFailure(Policy):
    reward = run_to_failure_cost
    discount = exponential_discount
    model1 = None

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: NDArray[np.float64],
        discount_rate: NDArray[np.float64],
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = model
        self.nb_assets = nb_assets
        self.cf = cf
        self.discount_rate = discount_rate
        self.model_args = model_args

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            run_to_failure_cost,
            exponential_discount,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discount_args=(self.discount_rate,),
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        f = (
            lambda x: run_to_failure_cost(x, self.cf)
            * exponential_discount.factor(x, self.discount_rate)
            / exponential_discount.annuity_factor(x, self.discount_rate)
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask, 0, self.model.ls_integrate(f, dt, timeline, *self.model_args)
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, dt: float = 1.0
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf), dt)

    def sample(
        self,
        nb_samples: int,
    ) -> RenewalRewardData:
        generator = lifetimes_rewards_generator(
            self.model,
            self.reward,
            self.discount,
            nb_samples,
            self.nb_assets,
            np.inf,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discount_args=(self.discount_rate,),
        )
        _lifetimes, _event_times, _total_rewards, _events, still_valid = next(generator)
        assets_index, samples_index = np.where(still_valid)
        assets_index.astype(np.int64)
        samples_index.astype(np.int64)
        lifetimes = _lifetimes[still_valid]
        event_times = _event_times[still_valid]
        total_rewards = _total_rewards[still_valid]
        events = _events[still_valid]
        order = np.zeros_like(lifetimes)

        return RenewalRewardData(
            samples_index,
            assets_index,
            order,
            event_times,
            lifetimes,
            events,
            self.model_args,
            False,
            total_rewards,
        )


class OneCycleAgeReplacementPolicy(Policy):
    reward = age_replacement_cost
    discount = exponential_discount
    model1 = None

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        discount_rate: NDArray[np.float64],
        *,
        ar: Optional[NDArray[np.float64]] = None,
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
    ) -> None:
        if a0 is not None:
            model = LeftTruncatedModel(model)
            model_args = (a0, *model_args)
        self.model = AgeReplacementModel(model)
        self.nb_assets = nb_assets

        self.model_args = model_args
        self.ar = ar
        self.cf = cf
        self.cp = cp
        self.discount_rate = discount_rate

    @ifset("ar")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return reward_partial_expectation(
            timeline,
            self.model,
            self.reward,
            self.discount,
            model_args=self.model_args,
            reward_args=(self.ar, self.cf, self.cp),
            discount_args=(self.discount_rate,),
        )

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.expected_total_cost(np.array(np.inf))

    @ifset("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64], dt: float = 1.0
    ) -> NDArray[np.float64]:
        f = (
            lambda x: age_replacement_cost(x, self.ar, self.cf, self.cp)
            * exponential_discount.factor(x, self.discount_rate)
            / exponential_discount.annuity_factor(x, self.discount_rate)
        )
        mask = timeline < dt
        q0 = self.model.cdf(dt, *self.model_args) * f(dt)
        return q0 + np.where(
            mask,
            0,
            self.model.ls_integrate(f, np.array(dt), timeline, *self.model_args),
        )

    def asymptotic_expected_equivalent_annual_cost(
        self, dt: float = 1.0
    ) -> NDArray[np.float64]:
        return self.expected_equivalent_annual_cost(np.array(np.inf), dt)

    @ifset("ar")
    def sample(self, nb_samples: int) -> RenewalRewardData:
        generator = lifetimes_rewards_generator(
            self.model,
            self.reward,
            self.discount,
            nb_samples,
            self.nb_assets,
            np.inf,
            model_args=self.model_args,
            reward_args=(self.ar, self.cf, self.cp),
            discount_args=(self.discount_rate,),
        )
        _lifetimes, _event_times, _total_rewards, _events, still_valid = next(generator)
        assets_index, samples_index = np.where(still_valid)
        assets_index.astype(np.int64)
        samples_index.astype(np.int64)
        lifetimes = _lifetimes[still_valid]
        event_times = _event_times[still_valid]
        total_rewards = _total_rewards[still_valid]
        events = _events[still_valid]
        order = np.zeros_like(lifetimes)

        return RenewalRewardData(
            samples_index,
            assets_index,
            order,
            event_times,
            lifetimes,
            events,
            self.model_args,
            False,
            total_rewards,
        )

    def fit(
        self,
        inplace: Optional[bool] = True,
    ) -> NDArray[np.float64]:

        cf_3d, cp_3d = np.array(self.cf, ndmin=3), np.array(self.cp, ndmin=3)
        x0 = np.minimum(np.sum(cp_3d, axis=0) / np.sum(cf_3d - cp_3d, axis=0), 1)
        if np.size(x0) == 1:
            x0 = np.tile(x0, (self.nb_assets, 1))

        def eq(a):
            return np.sum(
                self.discount.factor(a, self.discount_rate)
                / self.discount.annuity_factor(a, self.discount_rate)
                * (
                    (cf_3d - cp_3d) * self.model.hf(a, *self.model_args)
                    - cp_3d / self.discount.annuity_factor(a, self.discount_rate)
                ),
                axis=0,
            )

        ar = np.asarray(newton(eq, x0), dtype=np.float64)

        if inplace:
            self.ar = ar
            self.model_args = (ar,) + self.model_args[1:]

        return ar


class RunToFailure(Policy):
    """run to failure policy (facade object to an underlying renewal reward process)"""

    reward = run_to_failure_cost
    discount = exponential_discount

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: NDArray[np.float64],
        rate: NDArray[np.float64],
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
    ) -> None:

        if a0 is not None:
            if model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            model1 = LeftTruncatedModel(model)
            model1_args = (a0, *model_args)

        self.model = model
        self.model1 = model1

        self.model_args = model_args
        self.cf = cf
        self.rate = rate

        self.model_args = model_args
        self.model1_args = model1_args

        self.nb_assets = nb_assets

        self.rrp = RenewalRewardProcess(
            self.model,
            run_to_failure_cost,
            nb_assets=self.nb_assets,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discount_rate=self.rate,
            model1=self.model1,
            model1_args=self.model1_args,
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.rrp.expected_total_reward(timeline)

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        return self.rrp.asymptotic_expected_total_reward()

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.rrp.expected_equivalent_annual_cost(timeline)

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        return self.rrp.asymptotic_expected_equivalent_annual_cost()

    def sample(self, nb_samples: int, end_time: float) -> RenewalRewardData:
        return self.rrp.sample(nb_samples, end_time)

    def expected_number_of_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def expected_number_of_failures(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def expected_number_of_preventive_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


class AgeReplacementPolicy(Policy):

    reward = age_replacement_cost
    discount = exponential_discount

    def __init__(
        self,
        model: LifetimeModel[*ModelArgs],
        cf: NDArray[np.float64],
        cp: NDArray[np.float64],
        ar: NDArray[np.float64] = None,
        discount_rate: NDArray[np.float64] = 0,
        model_args: ModelArgs = (),
        nb_assets: int = 1,
        a0: Optional[NDArray[np.float64]] = None,
        model1: Optional[LifetimeModel[*Model1Args]] = None,
        model1_args: Model1Args = (),
    ) -> None:

        if a0 is not None:
            if model1 is not None:
                raise ValueError("model1 and a0 can't be set together")
            model1 = LeftTruncatedModel(model)
            model1_args = (a0, *model_args)

        self.model = model
        self.model1 = model1

        self.cf = cf
        self.cp = cp
        self.ar = ar

        self.discount_rate = discount_rate

        self.model_args = model_args
        self.model1_args = model1_args

        self.nb_assets = nb_assets

        self.rrp = RenewalRewardProcess(
            self.model,
            run_to_failure_cost,
            nb_assets=self.nb_assets,
            model_args=self.model_args,
            reward_args=(self.cf,),
            discount_rate=self.discount_rate,
            model1=self.model1,
            model1_args=self.model1_args,
        )

    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        pass

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        pass

    def expected_number_of_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def expected_number_of_failures(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...

    def expected_number_of_preventive_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]: ...


class TPolicy(Policy):
    discount = exponential_discount
    nb_assets: int = 1

    def __init__(
        self,
        nhpp: NHPP,
        c0: NDArray[np.float64],
        cr: NDArray[np.float64],
        rate: float,
        *,
        ar: Optional[NDArray[np.float64]] = None,
        nhpp_args: ModelArgs = (),
        nb_assets: int = 1,
    ) -> None:
        self.nhpp = nhpp
        self.nhpp_args = nhpp_args
        self.nb_assets = nb_assets
        self.rate = rate
        self.ar = ar
        self.c0 = c0
        self.cr = cr

    @ifset("ar")
    def expected_total_cost(self, timeline: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    def asymptotic_expected_total_cost(self) -> NDArray[np.float64]:
        pass

    @ifset("ar")
    def expected_equivalent_annual_cost(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def number_of_replacements(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def expected_number_of_repairs(
        self, timeline: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass

    def asymptotic_expected_equivalent_annual_cost(self) -> NDArray[np.float64]:
        pass

    def fit(
        self,
        inplace: bool = True,
    ) -> np.ndarray:
        x0 = self.nhpp.model.mean()

        cr_2d, c0_2d, *nhpp_args_2d = np.atleast_2d(self.cr, self.c0, *self.nhpp_args)
        if isinstance(nhpp_args_2d, np.ndarray):
            nhpp_args_2d = (nhpp_args_2d,)

        if self.rate != 0:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    (1 - np.exp(-self.rate * a))
                    / self.rate
                    * self.nhpp.intensity(a, *nhpp_args_2d)
                    - gauss_legendre(
                        lambda t: np.exp(-self.rate * t)
                        * self.nhpp.intensity(t, *nhpp_args_2d),
                        np.array(0.0),
                        a,
                        ndim=2,
                    )
                    - c0_2d / cr_2d
                )

        else:

            def dcost(a):
                a = np.atleast_2d(a)
                return (
                    a * self.nhpp.intensity(a, *nhpp_args_2d)
                    - self.nhpp.cumulative_intensity(a, *nhpp_args_2d)
                    - c0_2d / cr_2d
                )

        ar = newton(dcost, x0)

        ndim = max(map(np.ndim, (self.c0, self.cr, *self.nhpp_args)), default=0)
        if ndim < 2:
            ar = np.squeeze(ar)

        if inplace:
            self.ar = ar

        return ar
