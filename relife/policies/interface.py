import numpy as np
from numpy.typing import NDArray

from relife.core import LifetimeModel
from . import OneCycleRunToFailurePolicy
from .run_to_failure import DefaultRunToFailurePolicy
from .age_replacement import (
    DefaultAgeReplacementPolicy,
    NHPPAgeReplacementPolicy,
    OneCycleAgeReplacementPolicy,
)
from relife.types import TupleArrays


class RunToFailurePolicy:
    def __init__(
        self,
        model: LifetimeModel[*TupleArrays],
        costs: dict[NDArray[np.float64]],
        nature: str = "default",
        **kwargs,
    ):
        if nature == "default":
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
            self.policy = DefaultRunToFailurePolicy(
                model,
                cf,
                discounting_rate=discounting_rate,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
                model1=model1,
                model1_args=model1_args,
            )
        elif nature == "one_cycle":
            try:
                cf = costs["cf"]
            except KeyError:
                raise ValueError("Costs must only contain cf")
            discounting_rate = kwargs.get("discounting_rate", 0.0)
            model_args = kwargs.get("model_args", ())
            nb_assets = kwargs.get("nb_assets", 1)
            a0 = kwargs.get("a0", None)
            self.policy = OneCycleRunToFailurePolicy(
                model,
                cf,
                discounting_rate=discounting_rate,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
            )
        else:
            raise ValueError(
                f"Uncorrect nature {nature}, valid are : default, one_cycle or poisson"
            )


class AgeReplacementPolicy:
    def __init__(
        self,
        model: LifetimeModel[*TupleArrays],
        costs: dict[NDArray[np.float64]],
        nature: str = "default",
        **kwargs,
    ):
        if nature == "default":
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
            self.policy = DefaultAgeReplacementPolicy(
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
        elif nature == "one_cycle":
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
            self.policy = OneCycleAgeReplacementPolicy(
                model,
                cf,
                cp,
                discounting_rate=discounting_rate,
                ar=ar,
                model_args=model_args,
                nb_assets=nb_assets,
                a0=a0,
            )
        elif nature == "poisson":
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
            self.policy = NHPPAgeReplacementPolicy(
                model,
                cf,
                cr,
                discounting_rate=discounting_rate,
                ar=ar,
                model_args=model_args,
                nb_assets=nb_assets,
            )
        else:
            raise ValueError(
                f"Uncorrect nature {nature}, valid are : default, one_cycle or poisson"
            )

    def __getattr__(self, item):
        class_name = type(self).__name__
        if self.policy is None:
            raise ValueError("No policy as been loaded")
        if item in super().__getattribute__("policy"):
            return super().__getattribute__("policy")[item]
        raise AttributeError(f"{class_name} has no attribute/method {item}")


def policy(
    model: LifetimeModel[*TupleArrays],
    costs: dict[NDArray[np.float64]],
    age_replacement: bool = True,
    nature: str = "default",
    **kwargs,
):
    if age_replacement:
        if nature not in ("default", "one_cycle", "poisson"):
            raise ValueError(
                "Invalid age replacement policy nature. Valid values are : default, one_cycle or poisson"
            )
        return AgeReplacementPolicy(model, costs, nature=nature, **kwargs)
    else:
        if nature not in ("default", "one_cycle"):
            raise ValueError(
                "Invalid run to failure policy nature. Valid values are : default, one_cycle or poisson"
            )
        return RunToFailurePolicy(model, costs, nature=nature, **kwargs)
