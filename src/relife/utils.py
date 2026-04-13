from typing import Any, Literal, TypeAlias, overload

import numpy as np
from numpy.typing import NDArray
from optype.numpy import (
    Array,
    Array0D,
    Array1D,
    ArrayND,
    is_array_0d,
    is_array_1d,
)

__all__ = [
    "to_column_2d",
    "to_numpy_float",
    "flatten_if_at_least_2d",
    "get_args_nb_assets",
    "get_model_nb_assets",
]


@overload
def to_numpy_float(v: float | np.floating | np.uint) -> np.float64: ...
@overload
def to_numpy_float(
    v: ArrayND[np.floating | np.uint],
) -> ArrayND[np.float64]: ...
def to_numpy_float(
    v: float | np.uint | np.floating | ArrayND[np.floating | np.uint],
) -> np.float64 | ArrayND[np.float64]:
    """
    Convert the input to np.float64 if it is a scalar or an array of np.float64
    otherwise.
    """
    if isinstance(v, (int, float)):
        return np.float64(v)
    return np.asarray(v, dtype=np.float64)


ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint | np.bool


@overload
def to_column_2d(arg: ST | NumpyST) -> np.float64: ...
@overload
def to_column_2d(
    arg: Array0D[NumpyST],
) -> Array[tuple[Literal[1], Literal[1]], np.float64]: ...
@overload
def to_column_2d(
    arg: Array1D[NumpyST],
) -> Array[tuple[int, Literal[1]], np.float64]: ...
@overload
def to_column_2d(
    arg: Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> Array[tuple[int, int, *tuple[int, ...]], np.float64]: ...
def to_column_2d(
    arg: ST
    | NumpyST
    | Array0D[NumpyST]
    | Array1D[NumpyST]
    | Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> (
    np.float64
    | Array[tuple[Literal[1], Literal[1]], np.float64]
    | Array[tuple[int, Literal[1]], np.float64]
    | Array[tuple[int, int, *tuple[int, ...]], np.float64]
):
    """
    Reshapes ReLife arguments that are expected to be 0d or 1d.

    Parameters
    ----------
    arg : float, 0d or 1d array

    Returns
    -------
    np.float64 or (m, 1) shaped array
        Reshaped array used to ensure broadcasting compatibility in computations.
    """
    if isinstance(arg, np.ndarray):
        if arg.ndim <= 1:
            return arg.reshape(-1, 1).astype(np.float64)
        # typeguards
        assert not is_array_0d(arg) and not is_array_1d(arg)
        return arg.astype(np.float64)
    return np.float64(arg)


@overload
def flatten_if_at_least_2d(value: ST | NumpyST) -> np.float64: ...
@overload
def flatten_if_at_least_2d(value: Array0D[NumpyST]) -> Array0D[np.float64]: ...
@overload
def flatten_if_at_least_2d(value: Array1D[NumpyST]) -> Array1D[np.float64]: ...
@overload
def flatten_if_at_least_2d(
    value: Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> Array1D[np.float64]: ...
def flatten_if_at_least_2d(
    value: ST
    | NumpyST
    | Array0D[NumpyST]
    | Array1D[NumpyST]
    | Array[tuple[int, int, *tuple[int, ...]], NumpyST],
) -> np.float64 | Array0D[np.float64] | Array1D[np.float64]:
    """
    Flatten array-like object when possible.

    Parameters
    ----------
    value : np.ndarray

    Returns
    -------
    np.ndarray
        Flattened array.
    """
    if isinstance(value, np.ndarray):
        if value.ndim >= 1:
            return value.astype(np.float64).flatten()
        # typeguards
        assert is_array_0d(value)
        return value.astype(np.float64)
    return np.float64(value)


def get_args_nb_assets(*args: NDArray[Any]) -> int:
    """
    Gets the number of assets encoded in args.
    """
    if not bool(args):
        return 1
    reshaped_args = tuple(np.atleast_2d(arg) for arg in args)
    try:
        broadcast_shape = np.broadcast_shapes(*(ary.shape for ary in reshaped_args))
    except ValueError:
        raise ValueError("args have incompatible shapes")
    if len(broadcast_shape) == 0:
        return 1
    return broadcast_shape[0]


def get_model_nb_assets(model):
    """
    Gets the number of assets stored by a model (frozen or not).
    """
    from relife.base import FrozenParametricModel
    from relife.lifetime_model import EquilibriumDistribution, MinimumDistribution
    from relife.lifetime_model._regression import ParametricLifetimeRegression
    from relife.stochastic_process import NonHomogeneousPoissonProcess, RenewalProcess

    if isinstance(model, EquilibriumDistribution) or isinstance(
        model, MinimumDistribution
    ):
        return get_model_nb_assets(model.baseline)

    if isinstance(model, NonHomogeneousPoissonProcess):
        return get_model_nb_assets(model.lifetime_model)

    if isinstance(model, RenewalProcess):
        lifetime_model_nb_assets = get_model_nb_assets(model.lifetime_model)
        if model.first_lifetime_model is not None:
            first_lifetime_model_nb_assets = get_model_nb_assets(
                model.first_lifetime_model
            )
            return max(lifetime_model_nb_assets, first_lifetime_model_nb_assets)
        return lifetime_model_nb_assets

    if isinstance(model, FrozenParametricModel):
        if isinstance(model._unfrozen_model, NonHomogeneousPoissonProcess):
            return get_model_nb_assets(model._unfrozen_model)
        if isinstance(model._unfrozen_model, ParametricLifetimeRegression):
            # specific covar reshape
            reshaped_args = [np.atleast_2d(model.args[0])]
            reshaped_args += [to_column_2d(arg) for arg in model.args[1:]]
        else:
            reshaped_args = [to_column_2d(arg) for arg in model.args]
        return get_args_nb_assets(*reshaped_args)

    return 1
