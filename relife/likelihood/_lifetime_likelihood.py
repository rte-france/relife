# TODO : deplacer tout ce module dans lifetime_model._base
from __future__ import annotations

import copy
from abc import ABC
from typing import TYPE_CHECKING, Any, TypedDict, TypeVar, Unpack

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D, ToFloat
from typing_extensions import override

from relife.typing import ScipyMinimizeOptions
from relife.utils import reshape_1d_arg

from ._base import FittingResults, MaximumLikehoodOptimizer

__all__ = [
    "DefaultLifetimeLikelihood",
]

if TYPE_CHECKING:
    from relife.lifetime_model._base import FittableParametricLifetimeModel


M = TypeVar("M", bound=FittableParametricLifetimeModel[*tuple[Any, ...]])


# TODO : mettre dans lifetime_model._base (circular import)
class LifetimeData(TypedDict):
    complete_time: NDArray[np.float64]
    censored_time: NDArray[np.float64]  # 1d array or 2d
    left_truncations: NDArray[np.float64]
    complete_time_args: tuple[NDArray[Any], ...]
    censored_time_args: tuple[NDArray[Any], ...]
    left_truncations_args: tuple[NDArray[Any], ...]
    nb_observations: int


# TODO : mettre dans lifetime_model._base (circular import)
def _init_lifetime_data(
    time: NDArray[np.float64],  # 1d array or 2d
    model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
    event: NDArray[np.bool_] | None = None,
    entry: NDArray[np.float64] | None = None,
) -> LifetimeData:
    time = reshape_1d_arg(time)
    if time.shape[-1] == 2 and event is not None:
        raise ValueError("If time is given as intervals, event must be None")
    if time.shape[-1] == 1:
        event = (
            reshape_1d_arg(event)
            if event is not None
            else np.ones_like(time, dtype=np.bool_)
        )
    entry = (
        reshape_1d_arg(entry)
        if entry is not None
        else np.zeros(len(time), dtype=np.float64)
    )
    if np.any(time <= entry):
        raise ValueError("All time values must be greater than entry values")
    if isinstance(model_args, tuple):
        args = tuple((reshape_1d_arg(arg) for arg in model_args))
    elif isinstance(model_args, np.ndarray):
        args = (reshape_1d_arg(model_args),)
    elif model_args is None:
        args = ()
    sizes = [len(x) for x in (time, event, entry, *args) if x is not None]
    if len(set(sizes)) != 1:
        raise ValueError(
            f"""
            All lifetime data must have the same number of values. Fields
            length are different. Got {tuple(sizes)}
            """
        )
    non_zero_entry = np.flatnonzero(entry)
    if event is not None:
        non_zero_event = np.flatnonzero(event)
        data = LifetimeData(
            complete_time=time[non_zero_event],
            censored_time=time[~non_zero_event],
            left_truncations=entry[non_zero_entry],
            complete_time_args=tuple(arg[non_zero_event] for arg in args),
            censored_time_args=tuple(arg[~non_zero_event] for arg in args),
            left_truncations_args=tuple(arg[non_zero_entry] for arg in args),
            nb_observations=time.size,
        )
        return data

    complete_time_index = np.flatnonzero(time[:, 0] == time[:, 1])
    data = LifetimeData(
        complete_time=time[:, 1][complete_time_index],
        censored_time=time[~complete_time_index],
        left_truncations=entry[non_zero_entry],
        complete_time_args=tuple(arg[complete_time_index] for arg in args),
        censored_time_args=tuple(arg[~complete_time_index] for arg in args),
        left_truncations_args=tuple(arg[non_zero_entry] for arg in args),
        nb_observations=time.size,
    )
    return data


def _complete_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> float:
    if data["complete_time"].size == 0.0:
        return 0.0
    return -np.sum(
        np.log(model.pdf(data["complete_time"], *data["complete_time_args"]))
    )


def _jac_complete_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data["complete_time"].size == 0:
        return np.zeros_like(model.params)
    jac = -model.jac_pdf(
        data["complete_time"], *data["complete_time_args"]
    ) / model.pdf(data["complete_time"], *data["complete_time_args"])

    return np.sum(jac, axis=(1, 2))


def _censored_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> float:
    if data["censored_time"].size == 0:
        return 0.0
    if data["censored_time"].shape[-1] > 1:
        # interval censored time
        return np.sum(
            -np.log(
                10**-10
                + model.cdf(data["censored_time"][:, 1], *data["censored_time_args"])
                - model.cdf(data["censored_time"][:, 0], *data["censored_time_args"])
            ),
        )
    else:
        # right censored time
        return np.sum(model.chf(data["censored_time"], *data["censored_time_args"]))


def _jac_censored_time_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data["censored_time"].size == 0:
        return np.zeros_like(model.params)
    if data["censored_time"].shape[-1] > 1:
        # interval censored time
        jac_interval_censored = (
            model.jac_sf(data["censored_time"][:, 1], *data["censored_time_args"])
            - model.jac_sf(data["censored_time"][:, 0], *data["censored_time_args"])
        ) / (
            10**-10
            + model.cdf(data["censored_time"][:, 1], *data["censored_time_args"])
            - model.cdf(data["censored_time"][:, 0], *data["censored_time_args"])
        )

        return np.sum(jac_interval_censored, axis=(1, 2))
    else:
        # right censored time
        return np.sum(
            model.jac_chf(data["censored_time"], *data["censored_time_args"]), axis=1
        )


def _left_truncations_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> float:
    if data["left_truncations"].size == 0.0:
        return 0.0
    return -np.sum(model.chf(data["left_truncations"], *data["left_truncations_args"]))


def _jac_left_truncations_contrib(
    model: FittableParametricLifetimeModel[*tuple[Any, ...]],
    data: LifetimeData,
) -> NDArray[np.float64]:
    if data["left_truncations"].size == 0.0:
        return np.zeros_like(model.params)
    jac = -model.jac_chf(data["left_truncations"], *data["left_truncations_args"])
    return np.sum(jac, axis=1)


class DefaultLifetimeLikelihood(MaximumLikehoodOptimizer[M, LifetimeData], ABC):
    """
    Default likelihood from lifetime data.

    Parameters
    ----------
    model : generic FittableParametricLifetimeModel
        All model parameters must exist first. Its values are initialized by
        the likelihood with respect to data.
    # TODO
    """

    model: M
    data: LifetimeData

    def __init__(
        self,
        model: M,
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
    ):
        self.model = copy.deepcopy(model)
        self.data = _init_lifetime_data(
            time, model_args=model_args, event=event, entry=entry
        )

    @property
    @override
    def nb_observations(self) -> int:
        return self.data["nb_observations"]

    @override
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        self.model.params = params
        return (
            _complete_time_contrib(self.model, self.data)
            + _censored_time_contrib(self.model, self.data)
            + _left_truncations_contrib(self.model, self.data)
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
        self.model.params = params
        return (
            _jac_complete_time_contrib(self.model, self.data)
            + _jac_censored_time_contrib(self.model, self.data)
            + _jac_left_truncations_contrib(self.model, self.data)
        )

    @override
    def maximum_likelihood_estimation(
        self, **optimizer_options: Unpack[ScipyMinimizeOptions]
    ) -> FittingResults:
        if "jac" not in optimizer_options:
            optimizer_options["jac"] = self.jac_negative_log
        return super().maximum_likelihood_estimation(**optimizer_options)

        # hessian = approx_hessian(self, fitting_results.optimal_params)
        # fitting_results.covariance_matrix = np.linalg.pinv(hessian)
        # return fitting_results


# def _hessian_scheme(
#     likelihood: DefaultLifetimeLikelihood[M],
#     params: NDArray[np.float64],
#     method: Literal["2point", "cs"] = "cs",
#     eps: float = 1e-6,
# ) -> NDArray[np.float64]:
#     size = params.size
#     hess = np.empty((size, size))

#     # hessian 2 point
#     if method == "2point":
#         for i in range(size):
#             hess[i] = approx_fprime(
#                 params,
#                 lambda x: likelihood.jac_negative_log(x)[i],
#                 eps,
#             )
#         return hess
#     # hessian cs
#     u = eps * 1j * np.eye(size)
#     complex_params = params.astype(np.complex64)  # change params to complex
#     for i in range(size):
#         for j in range(i, size):
#             hess[i, j] = (
#                 np.imag(likelihood.jac_negative_log(complex_params + u[i])[j]) / eps
#             )
#             if i != j:
#                 hess[j, i] = hess[i, j]
#     return hess


# add approx_hessian str arg to options of fit instead of testing instance
# def approx_hessian(
#     likelihood: DefaultLifetimeLikelihood[M],
#     params: NDArray[np.float64],
#     eps: float = 1e-6,
# ) -> NDArray[np.float64]:
#     from relife.lifetime_model import Gamma
#     from relife.lifetime_model._parametric import ParametricLifetimeRegression

#     if isinstance(likelihood.model, ParametricLifetimeRegression):
#         if isinstance(likelihood.model.baseline, Gamma):
#             return _hessian_scheme(likelihood, params, method="2point", eps=eps)
#     if isinstance(likelihood.model, Gamma):
#         return _hessian_scheme(likelihood, params, method="2point", eps=eps)
#     return _hessian_scheme(likelihood, params, eps=eps)
