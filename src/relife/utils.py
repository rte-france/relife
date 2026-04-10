from typing import Any, Literal, TypeVar, overload

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array, Array0D, Array1D, Array2D, AtMost2D, is_array_2d

__all__ = [
    "to_2d_if_possible",
    "to_numpy_float",
    "flatten_if_possible",
    "get_args_nb_assets",
    "get_model_nb_assets",
]


@overload
def to_numpy_float(v: int | float) -> np.float64: ...
@overload
def to_numpy_float(v: Array[AtMost2D, np.float64]) -> Array[AtMost2D, np.float64]: ...
def to_numpy_float(
    v: int | float | Array[AtMost2D, np.float64],
) -> np.float64 | Array[AtMost2D, np.float64]:
    """
    Convert the input to np.float64 if it is a scalar or an array of np.float64
    otherwise.
    """
    if isinstance(v, (int, float)):
        return np.float64(v)
    return np.asarray(v, dtype=np.float64)


_T = TypeVar("_T", bound=np.generic)


@overload
def to_2d_if_possible(arg: int) -> np.float64: ...
@overload
def to_2d_if_possible(arg: float) -> np.float64: ...
@overload
def to_2d_if_possible(arg: Array0D[_T]) -> Array[tuple[Literal[1], Literal[1]], _T]: ...
@overload
def to_2d_if_possible(arg: Array1D[_T]) -> Array[tuple[int, Literal[1]], _T]: ...
@overload
def to_2d_if_possible(arg: Array2D[_T]) -> Array2D[_T]: ...
def to_2d_if_possible(
    arg: int | float | Array[AtMost2D, _T],
) -> np.float64 | np.bool_ | Array2D[_T]:
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
    if isinstance(arg, (int, float)):  # np.float64 is float
        return np.float64(arg)
    if arg.ndim == 1:
        return arg.reshape(-1, 1)
    if arg.ndim > 2:
        raise ValueError("arg can't be more than 2d")
    assert is_array_2d(arg)  # typeguards
    return arg


def flatten_if_possible(
    value: np.float64 | Array[AtMost2D, np.float64],
) -> np.float64 | Array[AtMost2D, np.float64]:
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
    if value.ndim != 0:
        return value.flatten()
    return value


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
            reshaped_args += [to_2d_if_possible(arg) for arg in model.args[1:]]
        else:
            reshaped_args = [to_2d_if_possible(arg) for arg in model.args]
        return get_args_nb_assets(*reshaped_args)

    return 1
