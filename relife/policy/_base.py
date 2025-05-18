from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Optional, overload

import numpy as np
from numpy.typing import NDArray

from relife import freeze
from relife.economic import Cost, ExponentialDiscounting, cost
from relife.lifetime_model import LeftTruncatedModel, FrozenLifetimeRegression, LifetimeDistribution, \
    FrozenAgeReplacementModel, FrozenLeftTruncatedModel, AgeReplacementModel

if TYPE_CHECKING:
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        LifetimeDistribution,
    )

    from .age_replacement import (
        AgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
    )
    from .run_to_failure import RunToFailurePolicy, OneCycleRunToFailurePolicy


# class AgeRenewalPolicy:
#
#     cost: Cost
#     model: LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel
#     model1: Optional[LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel]
#
#     def __init__(
#         self,
#         model: LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel,
#         model1: Optional[LifetimeDistribution | FrozenLifetimeRegression | FrozenAgeReplacementModel | FrozenLeftTruncatedModel] = None,
#         discounting_rate: float = 0.0,
#         **kwcost: float | NDArray[np.float64],
#     ):
#         self.model = model
#         self.model1 = model1
#         self.discounting = ExponentialDiscounting(discounting_rate)
#         self.cost = cost(**kwcost)
#
#     @property
#     def discounting_rate(self):
#         return self.discounting.rate

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
    #     use: str = "model",
    # ) -> tuple[NDArray[np.float64], ...]:
    #     from relife.sample import failure_data_sample
    #
    #     return failure_data_sample(self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use=use)


@overload
def run_to_failure_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    a0: Optional[float | NDArray[np.float64]] = None,
    one_cycle: Literal[True] = True,
    discounting_rate: float = 0.0,
) -> OneCycleRunToFailurePolicy: ...


@overload
def run_to_failure_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    a0: Optional[float | NDArray[np.float64]] = None,
    one_cycle: Literal[False] = False,
    discounting_rate: float = 0.0,
) -> RunToFailurePolicy: ...


def run_to_failure_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    a0: Optional[float | NDArray[np.float64]] = None,
    one_cycle: bool = False,
    discounting_rate: float = 0.0,
) -> RunToFailurePolicy | OneCycleRunToFailurePolicy:
    from .run_to_failure import RunToFailurePolicy, OneCycleRunToFailurePolicy

    discounting = ExponentialDiscounting(discounting_rate)
    if not one_cycle:
        first_lifetime_model = None
        if a0 is not None:
            first_lifetime_model : FrozenLeftTruncatedModel = freeze(LeftTruncatedModel(lifetime_model), a0)
        return RunToFailurePolicy(
            lifetime_model,
            cost,
            discounting,
            first_lifetime_model = first_lifetime_model,
        )
    if a0 is not None:
        lifetime_model : FrozenLeftTruncatedModel = freeze(LeftTruncatedModel(lifetime_model), a0)
    return OneCycleRunToFailurePolicy(
        lifetime_model,
        cost,
        discounting,
    )

@overload
def age_replacement_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    one_cycle = True,
    a0: Optional[float | NDArray[np.float64]] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1 = None,
    discounting_rate: float = 0.0,
) -> OneCycleAgeReplacementPolicy: ...


@overload
def age_replacement_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    one_cycle = False,
    a0: Optional[float | NDArray[np.float64]] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1: Optional[float | NDArray[np.float64]] = None,
    discounting_rate: float = 0.0,
) -> AgeReplacementPolicy: ...


def age_replacement_policy(
    lifetime_model: LifetimeDistribution | FrozenLifetimeRegression,
    cost: Cost,
    one_cycle: bool = False,
    a0: float | NDArray[np.float64] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1: Optional[float | NDArray[np.float64]] = None,

    discounting_rate: float = 0.0,
) -> OneCycleRunToFailurePolicy | RunToFailurePolicy:

    ar = np.nan if ar is None else ar
    if not one_cycle:
        first_lifetime_model = None
        if a0 is not None and ar1 is not None:
            first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar1, a0)
        elif ar1 is not None:
            first_lifetime_model = freeze(AgeReplacementModel(lifetime_model), ar1)
        elif a0 is not None:
            first_lifetime_model = freeze(AgeReplacementModel(LeftTruncatedModel(lifetime_model)), ar, a0)
        return AgeReplacementPolicy(
            freeze(AgeReplacementModel(lifetime_model), ar),
            cost,
            ExponentialDiscounting(discounting_rate),
            first_lifetime_model=first_lifetime_model
        )
    else:
        if ar1 is not None:
            raise ValueError
        if a0 is not None and



    return DefaultAgeReplacementPolicy(
        model,
        cost["cf"],
        cost["cp"],
        discounting_rate=discounting_rate,
        ar=ar,
        ar1=ar1,
        a0=a0,
        model1=model1,
    )


@overload
def run_to_failure_policy(
    model: LifetimeDistribution | FrozenParametricLifetimeModel,
    cost: Cost,
    one_cycle: Literal[False] = False,
    discounting_rate: Optional[float] = None,
    model1: Optional[LifetimeDistribution | FrozenParametricLifetimeModel] = None,
    a0: Optional[float | NDArray[np.float64]] = None,
) -> RunToFailurePolicy: ...



#
# def make_renewal_policy(
#     model: ParametricLifetimeModel[()] | NonHomogeneousPoissonProcess,
#     cost_structure: Cost,
#     one_cycle: bool = False,
#     run_to_failure: bool = False,
#     discounting_rate: Optional[float] = None,
#     model1: Optional[ParametricLifetimeModel[()] | NonHomogeneousPoissonProcess] = None,
#     a0: Optional[float | NDArray[np.float64]] = None,
#     ar: Optional[float | NDArray[np.float64]] = None,
#     ar1: Optional[float | NDArray[np.float64]] = None,
# ) -> RenewalPolicy:
#     """
#     Parameters
#     ----------
#     model : Parametric
#     cost_structure : dict of np.ndarray
#     one_cycle : bool, default False
#     run_to_failure : bool, default False
#     discounting_rate : float
#     ar1
#     ar
#     a0
#     model1
#     """
#
#     from relife.stochastic_process import NonHomogeneousPoissonProcess
#
#     from .age_replacement import (
#         DefaultAgeReplacementPolicy,
#         NonHomogeneousPoissonAgeReplacementPolicy,
#         OneCycleAgeReplacementPolicy,
#     )
#     from .run_to_failure import DefaultRunToFailurePolicy, OneCycleRunToFailurePolicy
#
#     if isinstance(model, NonHomogeneousPoissonProcess):
#         try:
#             cp, cr = (
#                 cost_structure["cp"],
#                 cost_structure["cr"],
#             )
#         except KeyError:
#             raise ValueError("Costs must contain cf and cr")
#         return NonHomogeneousPoissonAgeReplacementPolicy(
#             model,
#             cp,
#             cr,
#             discounting_rate=discounting_rate,
#             ar=ar,
#         )
#
#     if run_to_failure:
#         if not one_cycle:
#             try:
#                 cf = cost_structure["cf"]
#             except KeyError:
#                 raise ValueError("Costs must only contain cf")
#             return DefaultRunToFailurePolicy(
#                 model,
#                 cf,
#                 discounting_rate=discounting_rate,
#                 a0=a0,
#                 model1=model1,
#             )
#         else:
#             try:
#                 cf = cost_structure["cf"]
#             except KeyError:
#                 raise ValueError("Costs must only contain cf")
#             return OneCycleRunToFailurePolicy(
#                 model,
#                 cf,
#                 discounting_rate=discounting_rate,
#                 a0=a0,
#             )
#     else:
#         if not one_cycle:
#             try:
#                 cf, cp = (
#                     cost_structure["cf"],
#                     cost_structure["cp"],
#                 )
#             except KeyError:
#                 raise ValueError("Costs must contain cf and cp")
#             return DefaultAgeReplacementPolicy(
#                 model,
#                 cf,
#                 cp,
#                 discounting_rate=discounting_rate,
#                 ar=ar,
#                 ar1=ar1,
#                 a0=a0,
#                 model1=model1,
#             )
#         else:
#             try:
#                 cf, cp = (
#                     cost_structure["cf"],
#                     cost_structure["cp"],
#                 )
#             except KeyError:
#                 raise ValueError("Costs must contain cf and cp")
#             return OneCycleAgeReplacementPolicy(
#                 model,
#                 cf,
#                 cp,
#                 discounting_rate=discounting_rate,
#                 ar=ar,
#                 a0=a0,
#             )
