from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from relife.economic import CostStructure

from ..economic.discounting import exponential_discounting

from relife.sample import SampleFailureDataMixin, SampleMixin

if TYPE_CHECKING:
    from relife.model import BaseLifetimeModel, FrozenLifetimeModel
    from relife.stochastic_process import NonHomogeneousPoissonProcess


# RenewalPolicy
class RenewalPolicy(SampleMixin[()], SampleFailureDataMixin[()]):

    cost_structure: CostStructure
    model: FrozenLifetimeModel
    model1: Optional[FrozenLifetimeModel]
    nb_assets: int

    def __init__(
        self,
        model: BaseLifetimeModel[()],
        model1: Optional[BaseLifetimeModel[()]] = None,
        discounting_rate: Optional[float] = None,
        **kwcosts: float | NDArray[np.float64],
    ):
        from ..model._base import BaseDistribution

        if not model.frozen:
            raise ValueError
        if isinstance(model, BaseDistribution):
            model = model.freeze()
        if model1 is not None:
            if not model1.frozen:
                raise ValueError
            if isinstance(model1, BaseDistribution):
                model1 = model1.freeze()

        self.model = model
        self.model1 = model1
        self.discounting = exponential_discounting(discounting_rate)
        if self.model.nb_assets != self.model1.nb_assets:
            raise ValueError
        self.nb_assets = self.model.nb_assets
        self.cost_structure = CostStructure(**kwcosts)

        if self.cost_structure.nb_assets != self.nb_assets:
            raise ValueError("Given model args and costs differ in nb of assets")

    @property
    def discounting_rate(self):
        return self.discounting.rate

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
    #     return failure_data_sample(
    #         self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use=use
    #     )


def make_renewal_policy(
    model: BaseLifetimeModel[()] | NonHomogeneousPoissonProcess,
    cost_structure: CostStructure,
    one_cycle: bool = False,
    run_to_failure: bool = False,
    discounting_rate: Optional[float] = None,
    model1: Optional[BaseLifetimeModel[()] | NonHomogeneousPoissonProcess] = None,
    a0: Optional[float | NDArray[np.float64]] = None,
    ar: Optional[float | NDArray[np.float64]] = None,
    ar1: Optional[float | NDArray[np.float64]] = None,
) -> RenewalPolicy:
    """
    Parameters
    ----------
    model : Parametric
    cost_structure : dict of np.ndarray
    one_cycle : bool, default False
    run_to_failure : bool, default False
    discounting_rate : float
    ar1
    ar
    a0
    model1
    """

    from relife.stochastic_process import NonHomogeneousPoissonProcess

    from .age_replacement import (
        DefaultAgeReplacementPolicy,
        NonHomogeneousPoissonAgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
    )
    from .run_to_failure import DefaultRunToFailurePolicy, OneCycleRunToFailurePolicy

    if isinstance(model, NonHomogeneousPoissonProcess):
        try:
            cp, cr = (
                cost_structure["cp"],
                cost_structure["cr"],
            )
        except KeyError:
            raise ValueError("Costs must contain cf and cr")
        return NonHomogeneousPoissonAgeReplacementPolicy(
            model,
            cp,
            cr,
            discounting_rate=discounting_rate,
            ar=ar,
        )

    if run_to_failure:
        if not one_cycle:
            try:
                cf = cost_structure["cf"]
            except KeyError:
                raise ValueError("Costs must only contain cf")
            return DefaultRunToFailurePolicy(
                model,
                cf,
                discounting_rate=discounting_rate,
                a0=a0,
                model1=model1,
            )
        else:
            try:
                cf = cost_structure["cf"]
            except KeyError:
                raise ValueError("Costs must only contain cf")
            return OneCycleRunToFailurePolicy(
                model,
                cf,
                discounting_rate=discounting_rate,
                a0=a0,
            )
    else:
        if not one_cycle:
            try:
                cf, cp = (
                    cost_structure["cf"],
                    cost_structure["cp"],
                )
            except KeyError:
                raise ValueError("Costs must contain cf and cp")
            return DefaultAgeReplacementPolicy(
                model,
                cf,
                cp,
                discounting_rate=discounting_rate,
                ar=ar,
                ar1=ar1,
                a0=a0,
                model1=model1,
            )
        else:
            try:
                cf, cp = (
                    cost_structure["cf"],
                    cost_structure["cp"],
                )
            except KeyError:
                raise ValueError("Costs must contain cf and cp")
            return OneCycleAgeReplacementPolicy(
                model,
                cf,
                cp,
                discounting_rate=discounting_rate,
                ar=ar,
                a0=a0,
            )
