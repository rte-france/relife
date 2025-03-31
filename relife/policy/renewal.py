from __future__ import annotations

from typing import TYPE_CHECKING, Optional, NewType

import numpy as np
from numpy.typing import NDArray

from relife.economic import CostStructure
from relife.economic.rewards import (
    exponential_discounting,
)
from relife.model.protocol import LifetimeModel
from relife.sampling import CountData
from ..model.frozen import FrozenLifetimeModel
from ..parametric_model import Distribution

if TYPE_CHECKING:
    from .age_replacement import (
        DefaultAgeReplacementPolicy,
        NonHomogeneousPoissonAgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
    )
    from .run_to_failure import DefaultRunToFailurePolicy, OneCycleRunToFailurePolicy


Cost = NewType("Cost", NDArray[np.floating] | NDArray[np.integer] | float | int)


# RenewalPolicy
class RenewalPolicy:

    cost_structure: CostStructure
    model: FrozenLifetimeModel
    model1: Optional[FrozenLifetimeModel]

    def __init__(
        self,
        model: LifetimeModel[()],
        model1: Optional[LifetimeModel[()]] = None,
        discounting_rate: Optional[float] = None,
        **kwcosts: Cost,
    ):
        if not model.frozen:
            raise ValueError
        if isinstance(model, Distribution):
            model = model.freeze()
        if model1 is not None:
            if not model1.frozen:
                raise ValueError
            if isinstance(model1, Distribution):
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

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sampling import sample_count_data

        return sample_count_data(self, size, tf, t0=t0, maxsample=maxsample, seed=seed)

    def failure_data_sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
        use: str = "model",
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sampling import failure_data_sample

        return failure_data_sample(
            self, size, tf, t0=t0, maxsample=maxsample, seed=seed, use=use
        )


def make_renewal_policy(
    obj: Union[LifetimeModel[()], NonHomogeneousPoissonProcess],
    cost_structure: CostStructure,
    one_cycle: bool = False,
    run_to_failure: bool = False,
    discounting_rate: Optional[float] = None,
    **kwargs,
) -> RenewalPolicy:
    """
    Parameters
    ----------
    obj : Parametric
    cost_structure : dict of np.ndarray
    one_cycle : bool, default False
    run_to_failure : bool, default False
    discounting_rate : float
    nb_assets : int
    kwargs
    """

    from .age_replacement import (
        DefaultAgeReplacementPolicy,
        NonHomogeneousPoissonAgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
    )
    from .run_to_failure import DefaultRunToFailurePolicy, OneCycleRunToFailurePolicy
    from relife.stochastic_process import NonHomogeneousPoissonProcess

    if isinstance(obj, NonHomogeneousPoissonProcess):
        try:
            cp, cr = (
                cost_structure["cp"],
                cost_structure["cr"],
            )
        except KeyError:
            raise ValueError("Costs must contain cf and cr")
        ar = kwargs.get("ar", None)
        return NonHomogeneousPoissonAgeReplacementPolicy(
            obj,
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
            a0 = kwargs.get("a0", None)
            model1 = kwargs.get("model1", None)
            return DefaultRunToFailurePolicy(
                obj,
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
            model_args = kwargs.get("model_args", ())
            a0 = kwargs.get("a0", None)
            return OneCycleRunToFailurePolicy(
                obj,
                cf,
                discounting_rate=discounting_rate,
                model_args=model_args,
                nb_assets=nb_assets,
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
            ar = kwargs.get("ar", None)
            ar1 = kwargs.get("ar1", None)
            model_args = kwargs.get("model_args", ())
            a0 = kwargs.get("a0", None)
            model1 = kwargs.get("model1", None)
            model1_args = kwargs.get("model1_args", None)
            return DefaultAgeReplacementPolicy(
                obj,
                cf,
                cp,
                discounting_rate=discounting_rate,
                ar=ar,
                ar1=ar1,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
                model1=model1,
                model1_args=model1_args,
            )
        else:
            try:
                cf, cp = (
                    cost_structure["cf"],
                    cost_structure["cp"],
                )
            except KeyError:
                raise ValueError("Costs must contain cf and cp")
            ar = kwargs.get("ar", None)
            model_args = kwargs.get("model_args", ())
            a0 = kwargs.get("a0", None)
            return OneCycleAgeReplacementPolicy(
                obj,
                cf,
                cp,
                discounting_rate=discounting_rate,
                ar=ar,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
            )
