from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np
from numpy.typing import NDArray

from relife.core import ParametricModel

if TYPE_CHECKING:
    from relife.policies import (
        DefaultAgeReplacementPolicy,
        DefaultRunToFailurePolicy,
        NonHomogeneousPoissonAgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
        OneCycleRunToFailurePolicy,
    )


def renewal_policy(
    obj: ParametricModel,
    costs: dict[NDArray[np.float64]],
    one_cycle: bool = False,
    run_to_failure: bool = False,
    discounting_rate: Optional[float] = None,
    nb_assets: int = 1,
    **kwargs,
) -> Union[
    NonHomogeneousPoissonAgeReplacementPolicy,
    DefaultAgeReplacementPolicy,
    DefaultRunToFailurePolicy,
    OneCycleRunToFailurePolicy,
    OneCycleAgeReplacementPolicy,
]:
    """
    Parameters
    ----------
    obj : ParametricModel
    costs : dict of np.ndarray
    one_cycle : bool, default False
    run_to_failure : bool, default False
    discounting_rate : float
    nb_assets : int
    kwargs
    """

    from relife.policies import (
        DefaultAgeReplacementPolicy,
        DefaultRunToFailurePolicy,
        NonHomogeneousPoissonAgeReplacementPolicy,
        OneCycleAgeReplacementPolicy,
        OneCycleRunToFailurePolicy,
    )
    from relife.process import NonHomogeneousPoissonProcess

    if isinstance(obj, NonHomogeneousPoissonProcess):
        try:
            cp, cr = (
                costs["cp"],
                costs["cr"],
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
            nb_assets=nb_assets,
        )

    if run_to_failure:
        if not one_cycle:
            try:
                cf = costs["cf"]
            except KeyError:
                raise ValueError("Costs must only contain cf")
            model_args = kwargs.get("model_args", ())
            a0 = kwargs.get("a0", None)
            model1 = kwargs.get("model1", None)
            model1_args = kwargs.get("model1_args", None)
            return DefaultRunToFailurePolicy(
                obj,
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
                    costs["cf"],
                    costs["cp"],
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
                    costs["cf"],
                    costs["cp"],
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
