from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias, final

import numpy as np
from optype.numpy import Array, Array1D, ArrayND
from scipy.optimize import Bounds
from typing_extensions import override

from relife.base import FitConfig, MaximumLikelihoodOptimizer
from relife.lifetime_models import (
    Gamma,
    Gompertz,
    LifetimeDistribution,
    MinimumDistribution,
    ParametricLifetimeRegression,
)
from relife.utils import to_column_2d_if_1d

__all__ = ["LifetimeLikelihood"]


@dataclass
class LifetimeData:
    nb_observations: int = field(init=False)
    complete_time: Array[tuple[int, Literal[1]], np.float64] = field(
        init=False, repr=False
    )
    censored_time: (
        Array[tuple[int, Literal[1]], np.float64]
        | Array[tuple[int, Literal[2]], np.float64]
    ) = field(init=False, repr=False)
    left_truncations: Array[tuple[int, Literal[1]], np.float64] = field(
        init=False, repr=False
    )
    complete_time_args: tuple[Array[tuple[int, Literal[1]], np.float64], ...] = field(
        init=False, repr=False
    )
    censored_time_args: tuple[Array[tuple[int, Literal[1]], np.float64], ...] = field(
        init=False, repr=False
    )
    left_truncations_args: tuple[Array[tuple[int, Literal[1]], np.float64], ...] = (
        field(init=False, repr=False)
    )

    def __init__(
        self,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        args: Sequence[Array1D[np.float64]] = (),
    ) -> None:
        column_time = to_column_2d_if_1d(time)
        if column_time.shape[-1] == 2 and event is not None:
            raise ValueError("If time is given as intervals, event must be None")
        column_event = None
        if column_time.shape[-1] == 1:
            column_event = (
                to_column_2d_if_1d(event)
                if event is not None
                else np.ones_like(time, dtype=np.bool_)
            )
        column_entry = (
            to_column_2d_if_1d(entry)
            if entry is not None
            else np.zeros(len(time), dtype=np.float64)
        )
        if np.any(column_time <= column_entry):
            raise ValueError("All time values must be greater than entry values")
        column_args = tuple(to_column_2d_if_1d(arg) for arg in args)
        sizes = [
            len(x)
            for x in (column_time, column_event, column_entry, *column_args)
            if x is not None
        ]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}
                """
            )
        non_zero_entry = np.flatnonzero(column_entry)
        if column_event is not None:
            non_zero_event = np.flatnonzero(column_event)
            zero_event = np.flatnonzero(column_event == 0)
            self.nb_observations = len(time)
            self.complete_time = column_time[non_zero_event]
            self.censored_time = column_time[zero_event]
            self.left_truncations = column_entry[non_zero_entry]
            self.complete_time_args = tuple(arg[non_zero_event] for arg in column_args)
            self.censored_time_args = tuple(arg[zero_event] for arg in column_args)
            self.left_truncations_args = tuple(
                arg[non_zero_entry] for arg in column_args
            )
        else:
            complete_time_index = np.flatnonzero(column_time[:, 0] == column_time[:, 1])
            non_complete_time_index = np.flatnonzero(
                column_time[:, 0] != column_time[:, 1]
            )
            self.nb_observations = len(time)
            self.complete_time = column_time[:, 1][complete_time_index]
            self.censored_time = column_time[non_complete_time_index]
            self.left_truncations = column_entry[non_zero_entry]
            self.complete_time_args = tuple(
                arg[complete_time_index] for arg in column_args
            )
            self.censored_time_args = tuple(
                arg[non_complete_time_index] for arg in column_args
            )
            self.left_truncations_args = tuple(
                arg[non_zero_entry] for arg in column_args
            )


FittableParametricLifetimeModel: TypeAlias = (
    LifetimeDistribution | ParametricLifetimeRegression | MinimumDistribution
)


@final
class LifetimeLikelihood(
    MaximumLikelihoodOptimizer[FittableParametricLifetimeModel, LifetimeData]
):
    """
    Maximum likelihood estimator from lifetime data.

    Parameters
    ----------
    model : generic FittableParametricLifetimeModel
        Every model parameters must be initialized before passing it to the
        likelihood.
    data : LifetimeData
        An object that encapsulate and preprocess lifetime observations and
        truncations.
    config : OptimizerConfig
        An object that groups configurations used by the optimizer.

    Attributes
    ----------
    model: FittableParametricLifetimeModel
        A copy of the original model.
    data : LifetimeData
        An object that encapsulate and preprocess lifetime observations and
        truncations.
    config : OptimizerConfig
        An object that groups configurations used by the optimizer.
    """

    data: LifetimeData

    def __init__(
        self,
        model: FittableParametricLifetimeModel,
        data: LifetimeData,
        config: FitConfig,
    ):
        self.model = model
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return self.data.nb_observations

    @override
    def negative_log(self, params: Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return (
            complete_time_contrib(self.model, self.data)
            + censored_time_contrib(self.model, self.data)
            + left_truncations_contrib(self.model, self.data)
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The jacobian is computed with respect to parameters.

        Parameters
        ----------
        model : parametric model
            A parametrized model with appropriate parameters values.

        Returns
        -------
        out : ndarray
        """
        self.model.set_params(params)
        return (
            jac_complete_time_contrib(self.model, self.data)
            + jac_censored_time_contrib(self.model, self.data)
            + jac_left_truncations_contrib(self.model, self.data)
        )

    @classmethod
    def from_data(
        cls: type["LifetimeLikelihood"],
        model: FittableParametricLifetimeModel,
        time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
        args: Sequence[Array1D[np.float64]] = (),
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> "LifetimeLikelihood":
        if isinstance(model, LifetimeDistribution):
            return init_likelihood_from_distribution(
                model, time, event, entry, **kwargs
            )
        if isinstance(model, ParametricLifetimeRegression):
            return init_likelihood_from_regression(
                model, time, args, event, entry, **kwargs
            )
        return init_likelihood_from_minimum_distribution(
            model, time, args, event, entry, **kwargs
        )


def init_likelihood_from_distribution(
    model: LifetimeDistribution,
    time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
    event: Array1D[np.bool_] | None = None,
    entry: Array1D[np.float64] | None = None,
    **kwargs: Any,
) -> LifetimeLikelihood:
    lifetime_data = LifetimeData(time, event=event, entry=entry)
    fresh_distrib = type(model)()
    x0 = kwargs.get(
        "x0", init_distrib_params_from_lifetimes(fresh_distrib, lifetime_data)
    )
    config = FitConfig(x0)
    config.scipy_minimize_options["bounds"] = kwargs.get(
        "bounds", get_distrib_params_bounds(fresh_distrib)
    )
    config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
    config.covariance_method = kwargs.get(
        "covariance_method", "2point" if isinstance(fresh_distrib, Gamma) else "cs"
    )
    return LifetimeLikelihood(fresh_distrib, lifetime_data, config)


def init_likelihood_from_regression(
    model: ParametricLifetimeRegression,
    time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
    covar: Sequence[Array1D[np.float64]],
    event: Array1D[np.bool_] | None = None,
    entry: Array1D[np.float64] | None = None,
    **kwargs: Any,
) -> LifetimeLikelihood:
    fresh_regression = type(model)(
        type(model.baseline)(), coefficients=(0.0,) * len(covar)
    )  # init new regression object with appropriate number of covar
    lifetime_data = LifetimeData(time, event, entry, covar)
    x0 = kwargs.get(
        "x0", init_regression_params_from_lifetimes(fresh_regression, lifetime_data)
    )
    fresh_regression.set_params(x0)
    config = FitConfig(x0)
    config.scipy_minimize_options["bounds"] = kwargs.get(
        "bounds", get_regression_params_bounds(fresh_regression)
    )
    config.scipy_minimize_options["method"] = kwargs.get("method", "L-BFGS-B")
    config.covariance_method = kwargs.get(
        "covariance_method",
        "2point" if isinstance(fresh_regression.baseline, Gamma) else "cs",
    )
    return LifetimeLikelihood(fresh_regression, lifetime_data, config)


def init_likelihood_from_minimum_distribution(
    model: MinimumDistribution,
    time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64],
    args: Sequence[Array1D[np.float64]] = (),
    event: Array1D[np.bool_] | None = None,
    entry: Array1D[np.float64] | None = None,
    **kwargs: Any,
) -> LifetimeLikelihood:
    if isinstance(model.baseline, LifetimeDistribution):
        likelihood = init_likelihood_from_distribution(
            model.baseline, time, event, entry, **kwargs
        )
    else:
        likelihood = init_likelihood_from_regression(
            model.baseline, time, args, event, entry, **kwargs
        )
    assert isinstance(
        likelihood.model, (LifetimeDistribution, ParametricLifetimeRegression)
    )
    likelihood.model = MinimumDistribution(likelihood.model, model.n)
    return likelihood


def complete_time_contrib(
    model: FittableParametricLifetimeModel,
    data: LifetimeData,
) -> float:
    if data.complete_time.size == 0.0:
        return 0.0
    res = -np.sum(np.log(model.pdf(data.complete_time, *data.complete_time_args)))
    return res


def jac_complete_time_contrib(
    model: FittableParametricLifetimeModel,
    data: LifetimeData,
) -> ArrayND[np.float64]:
    if data.complete_time.size == 0:
        return np.zeros_like(model.get_params())
    jac = -model.jac_pdf(data.complete_time, *data.complete_time_args) / model.pdf(
        data.complete_time, *data.complete_time_args
    )

    return np.sum(jac, axis=(1, 2))


def censored_time_contrib(
    model: FittableParametricLifetimeModel,
    data: LifetimeData,
) -> float:
    if data.censored_time.size == 0:
        return 0.0
    if data.censored_time.shape[-1] > 1:
        # interval censored time
        return np.sum(
            -np.log(
                10**-10
                + model.cdf(data.censored_time[:, 1], *data.censored_time_args)
                - model.cdf(data.censored_time[:, 0], *data.censored_time_args)
            ),
        )
    else:
        # right censored time
        return np.sum(model.chf(data.censored_time, *data.censored_time_args))


def jac_censored_time_contrib(
    model: FittableParametricLifetimeModel,
    data: LifetimeData,
) -> ArrayND[np.float64]:
    if data.censored_time.size == 0:
        return np.zeros_like(model.get_params())
    if data.censored_time.shape[-1] > 1:
        # interval censored time
        jac_interval_censored = (
            model.jac_sf(data.censored_time[:, 1], *data.censored_time_args)
            - model.jac_sf(data.censored_time[:, 0], *data.censored_time_args)
        ) / (
            10**-10
            + model.cdf(data.censored_time[:, 1], *data.censored_time_args)
            - model.cdf(data.censored_time[:, 0], *data.censored_time_args)
        )

        return np.sum(jac_interval_censored, axis=(1, 2))
    else:
        # right censored time
        return np.sum(
            model.jac_chf(data.censored_time, *data.censored_time_args),
            axis=(1, 2),
        )


def left_truncations_contrib(
    model: FittableParametricLifetimeModel,
    data: LifetimeData,
) -> float:
    if data.left_truncations.size == 0.0:
        return 0.0
    return -np.sum(model.chf(data.left_truncations, *data.left_truncations_args))


def jac_left_truncations_contrib(
    model: FittableParametricLifetimeModel,
    data: LifetimeData,
) -> ArrayND[np.float64]:
    if data.left_truncations.size == 0.0:
        return np.zeros_like(model.get_params())
    jac = -model.jac_chf(data.left_truncations, *data.left_truncations_args)
    return np.sum(jac, axis=(1, 2))


def init_distrib_params_from_lifetimes(
    model: LifetimeDistribution, data: LifetimeData
) -> Array1D[np.float64]:
    # flatten censored_time in case it is 2D
    all_time_values = np.concatenate(
        (data.complete_time.flatten(), data.censored_time.flatten())
    )
    nb_params = model.get_params().size
    if isinstance(model, Gompertz):
        param0 = np.empty(nb_params, dtype=np.float64)
        rate = np.pi / (np.sqrt(6) * np.std(all_time_values))
        shape = np.exp(-rate * np.mean(all_time_values))
        param0[0] = shape
        param0[1] = rate
        return param0

    param0 = np.ones(nb_params, dtype=np.float64)
    param0[-1] = 1 / np.median(all_time_values)
    return param0


def init_regression_params_from_lifetimes(
    model: ParametricLifetimeRegression, data: LifetimeData
) -> Array1D[np.float64]:
    param0 = np.zeros_like(model.get_params(), dtype=np.float64)
    param0[-model.baseline.get_params().size :] = init_distrib_params_from_lifetimes(
        model.baseline, data
    )
    return param0


def get_distrib_params_bounds(model: LifetimeDistribution) -> Bounds:
    nb_params = model.get_params().size
    return Bounds(
        np.full(nb_params, np.finfo(float).resolution),
        np.full(nb_params, np.inf),
    )


def get_regression_params_bounds(model: ParametricLifetimeRegression) -> Bounds:
    nb_coefficients = model.covar_effect.get_params().size
    lb = np.concatenate(
        (
            np.full(nb_coefficients, -np.inf),
            get_distrib_params_bounds(
                model.baseline
            ).lb,  # baseline has _params_bounds according to typing
        )
    )
    ub = np.concatenate(
        (
            np.full(nb_coefficients, np.inf),
            get_distrib_params_bounds(model.baseline).ub,
        )
    )
    return Bounds(lb, ub)
