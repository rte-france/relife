from typing import Optional

import numpy as np
from numpy.typing import NDArray

from relife.core import LifetimeModel
from relife.core.descriptors import NbAssets, ShapedArgs
from relife.data import CountData
from relife.types import Arg


class ReplacementPolicy:
    model: LifetimeModel[*tuple[Arg, ...]]
    model1 = Optional[LifetimeModel[*tuple[Arg, ...]]]
    model_args = ShapedArgs(astuple=True)
    model1_args = ShapedArgs(astuple=True)
    nb_assets = NbAssets()

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

    def sample_lifetime_data(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        seed: Optional[int] = None,
        use: str = "model",
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sampling import sample_lifetime_data

        return sample_lifetime_data(self, size, tf, t0, seed, use)


def replacement_policy(
    model: LifetimeModel[*tuple[Arg, ...]],
    costs: dict[NDArray[np.float64]],
    one_cycle: bool = False,
    run_to_failure: bool = False,
    **kwargs,
) -> ReplacementPolicy:

    from .age_replacement import (
        OneCycleAgeReplacementPolicy,
        DefaultAgeReplacementPolicy,
    )
    from .run_to_failure import OneCycleRunToFailurePolicy, DefaultRunToFailurePolicy

    if run_to_failure:
        if not one_cycle:
            try:
                cf = costs["cf"]
            except KeyError:
                raise ValueError("Costs must only contain cf")
            discounting_rate = kwargs.get("discounting_rate", 0.0)
            model_args = kwargs.get("model_args", ())
            nb_assets = kwargs.get("nb_assets", 1)
            a0 = kwargs.get("a0", None)
            model1 = kwargs.get("model1", None)
            model1_args = kwargs.get("model1_args", None)
            return DefaultRunToFailurePolicy(
                model,
                cf,
                discounting_rate=discounting_rate,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
                model1=model1,
                model1_args=model1_args,
            )
        else:
            try:
                cf = costs["cf"]
            except KeyError:
                raise ValueError("Costs must only contain cf")
            discounting_rate = kwargs.get("discounting_rate", 0.0)
            model_args = kwargs.get("model_args", ())
            nb_assets = kwargs.get("nb_assets", 1)
            a0 = kwargs.get("a0", None)
            return OneCycleRunToFailurePolicy(
                model,
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
                    costs["cf"],
                    costs["cp"],
                )
            except KeyError:
                raise ValueError("Costs must contain cf and cp")
            discounting_rate = kwargs.get("discounting_rate", 0.0)
            ar = kwargs.get("ar", None)
            ar1 = kwargs.get("ar1", None)
            model_args = kwargs.get("model_args", ())
            nb_assets = kwargs.get("nb_assets", 1)
            a0 = kwargs.get("a0", None)
            model1 = kwargs.get("model1", None)
            model1_args = kwargs.get("model1_args", None)
            return DefaultAgeReplacementPolicy(
                model,
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
                    costs["cf"],
                    costs["cp"],
                )
            except KeyError:
                raise ValueError("Costs must contain cf and cp")
            discounting_rate = kwargs.get("discounting_rate", 0.0)
            ar = kwargs.get("ar", None)
            model_args = kwargs.get("model_args", ())
            nb_assets = kwargs.get("nb_assets", 1)
            a0 = kwargs.get("a0", None)
            return OneCycleAgeReplacementPolicy(
                model,
                cf,
                cp,
                discounting_rate=discounting_rate,
                ar=ar,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
            )


def unperfect_repair_policy(
    model: LifetimeModel[*tuple[Arg, ...]],
    costs: dict[NDArray[np.float64]],
    **kwargs,
):

    from .age_replacement import NonHomogeneousPoissonAgeReplacementPolicy

    try:
        cf, cr = (
            costs["cf"],
            costs["cr"],
        )
    except KeyError:
        raise ValueError("Costs must contain cf and cr")
    discounting_rate = kwargs.get("discounting_rate", 0.0)
    ar = kwargs.get("ar", None)
    model_args = kwargs.get("model_args", ())
    nb_assets = kwargs.get("nb_assets", 1)
    return NonHomogeneousPoissonAgeReplacementPolicy(
        model,
        cf,
        cr,
        discounting_rate=discounting_rate,
        ar=ar,
        model_args=model_args,
        nb_assets=nb_assets,
    )
