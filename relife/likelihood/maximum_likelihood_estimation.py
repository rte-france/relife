from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, TypeVarTuple, overload

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize

from relife import ParametricModel
from relife.data import LifetimeData, NHPPData
from relife.lifetime_model import (
    LifetimeDistribution,
    LifetimeRegression,
    ParametricLifetimeModel,
)
from relife.stochastic_process import NonHomogeneousPoissonProcess

from .lifetime_likelihood import LikelihoodFromLifetimes

Args = TypeVarTuple("Args")


def get_params_bounds(model: LifetimeDistribution | LifetimeRegression[*Args]) -> Bounds:
    from relife.lifetime_model import LifetimeDistribution, LifetimeRegression

    match model:
        case LifetimeDistribution():
            return Bounds(
                np.full(model.nb_params, np.finfo(float).resolution),
                np.full(model.nb_params, np.inf),
            )

        case LifetimeRegression():
            model: LifetimeRegression[*Args]
            lb = np.concatenate(
                (
                    np.full(model.covar_effect.nb_params, -np.inf),
                    get_params_bounds(model.baseline).lb,
                )
            )
            ub = np.concatenate(
                (
                    np.full(model.covar_effect.nb_params, np.inf),
                    get_params_bounds(model.baseline).ub,
                )
            )
            return Bounds(lb, ub)

        case _:
            raise NotImplemented


def init_params_from_lifetimes(
    model: LifetimeDistribution | LifetimeRegression[*Args],
    lifetime_data: LifetimeData,
) -> NDArray[np.float64]:
    from relife.lifetime_model import (
        AcceleratedFailureTime,
        Exponential,
        Gamma,
        Gompertz,
        LifetimeDistribution,
        LifetimeRegression,
        LogLogistic,
        ProportionalHazard,
        Weibull,
    )

    match model:

        case Exponential() | Weibull() | LogLogistic() | Gamma():
            model: LifetimeDistribution
            param0 = np.ones(model.nb_params, dtype=np.float64)
            param0[-1] = 1 / np.median(lifetime_data.complete_or_right_censored.values)
            return param0

        case Gompertz():
            model: LifetimeDistribution
            param0 = np.empty(model.nb_params, dtype=np.float64)
            rate = np.pi / (np.sqrt(6) * np.std(lifetime_data.complete_or_right_censored.values))
            shape = np.exp(-rate * np.mean(lifetime_data.complete_or_right_censored.values))
            param0[0] = shape
            param0[1] = rate
            return param0

        case ProportionalHazard() | AcceleratedFailureTime():
            model: LifetimeRegression
            baseline_param0 = init_params_from_lifetimes(model.baseline, lifetime_data)
            param0 = np.zeros_like(model.params, dtype=np.float64)
            param0[-baseline_param0.size :] = baseline_param0
            return param0

        case _:
            raise NotImplemented


# FailureData can be any of the union LifetimeData | NHPPData
FailureData = TypeVar("FailureData", bound=LifetimeData | NHPPData)


@overload
def maximum_likelihood_estimation(
    model: LifetimeDistribution, data: LifetimeData, **kwargs: Any
) -> LifetimeDistribution: ...


@overload
def maximum_likelihood_estimation(
    model: LifetimeRegression[*Args], data: LifetimeData, **kwargs: Any
) -> LifetimeRegression[*Args]: ...


@overload
def maximum_likelihood_estimation(
    model: NonHomogeneousPoissonProcess[*Args], data: NHPPData, **kwargs: Any
) -> NonHomogeneousPoissonProcess[*Args]: ...


def maximum_likelihood_estimation(
    model: LifetimeDistribution | LifetimeRegression[*Args] | NonHomogeneousPoissonProcess[*Args],
    data: LifetimeData | NHPPData,
    **kwargs: Any,
) -> LifetimeDistribution | LifetimeRegression[*Args] | NonHomogeneousPoissonProcess[*Args]:
    match model:
        case LifetimeDistribution() | LifetimeRegression():
            from relife import FittingResults
            from relife.data import LifetimeData

            if not isinstance(data, LifetimeData):
                raise ValueError

            # Step 2: Initialize the model and likelihood
            model.params = init_params_from_lifetimes(model, data)
            print("params names", model.params_names)
            print("params0", model.params)
            likelihood = LikelihoodFromLifetimes(model, data)

            try:
                bounds = get_params_bounds(model)
            except NotImplemented:
                bounds = None

            # Step 3: Configure and run the optimizer
            minimize_kwargs = {
                "method": kwargs.get("method", "L-BFGS-B"),
                "constraints": kwargs.get("constraints", ()),
                "bounds": kwargs.get("bounds", bounds),
                "x0": kwargs.get("x0", model.params),
            }
            print("first_jac", likelihood.jac_negative_log(model.params))
            optimizer = minimize(
                likelihood.negative_log,
                minimize_kwargs.pop("x0"),
                jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
                callback=lambda x: print(x),
                **minimize_kwargs,
            )
            print("fitted params", optimizer.x)
            # Step 4: Compute parameters variance (Hessian inverse)
            hessian_inverse = np.linalg.inv(likelihood.hessian())
            model.fitting_results = FittingResults(len(data), optimizer, var=hessian_inverse)
            model.params = optimizer.x
            return model

        case NonHomogeneousPoissonProcess():
            model: NonHomogeneousPoissonProcess[*tuple[float | NDArray[np.float64], ...]]

            if not isinstance(data, NHPPData):
                raise ValueError

            lifetime_data = data.to_lifetime_data()

            return maximum_likelihood_estimation(model.baseline, lifetime_data, **kwargs)

        case _:
            raise NotImplemented
