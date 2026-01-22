# pyright: basic

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVarTuple, overload

import numpy as np

from ._array_api import get_args_nb_assets, reshape_1d_arg

__all__ = ["is_frozen", "is_lifetime_model", "is_non_homogeneous_poisson_process", "get_model_nb_assets"]


if TYPE_CHECKING:
    from relife.base import FrozenParametricModel, ParametricModel
    from relife.lifetime_model._base import ParametricLifetimeModel
    from relife.lifetime_model._frozen import FrozenParametricLifetimeModel


Ts = TypeVarTuple("Ts")


@overload
def is_frozen(model: FrozenParametricModel[ParametricModel, *Ts]) -> Literal[True]: ...
@overload
def is_frozen(model: ParametricModel | FrozenParametricModel[ParametricModel, *Ts]) -> bool: ...
def is_frozen(model: ParametricModel | FrozenParametricModel[ParametricModel, *Ts]) -> bool:
    """
    Checks if model is frozen
    """
    from relife.base import FrozenParametricModel

    return isinstance(model, FrozenParametricModel)


@overload
def is_lifetime_model(model: FrozenParametricLifetimeModel[*Ts]) -> Literal[True]: ...
@overload
def is_lifetime_model(
    model: ParametricLifetimeModel[*Ts] | FrozenParametricLifetimeModel[ParametricModel, *Ts],
) -> Literal[True]: ...
@overload
def is_lifetime_model(
    model: Any | ParametricLifetimeModel[*Ts] | FrozenParametricLifetimeModel[ParametricModel, *Ts],
) -> bool: ...
def is_lifetime_model(
    model: Any | ParametricLifetimeModel[*Ts] | FrozenParametricLifetimeModel[ParametricModel, *Ts],
) -> bool:
    """
    Checks if model is a lifetime model.
    """
    # local import to avoid circular import
    from relife.lifetime_model._base import ParametricLifetimeModel

    if is_frozen(model):
        return isinstance(model.unfrozen_model, ParametricLifetimeModel)
    return isinstance(model, ParametricLifetimeModel)


def is_non_homogeneous_poisson_process(model):
    """
    Checks if model is a non-homogeneous Poisson process.
    """
    # local import to avoid circular import
    from relife.stochastic_process import NonHomogeneousPoissonProcess

    if is_frozen(model):
        return isinstance(model.unfrozen_model, NonHomogeneousPoissonProcess)
    return isinstance(model, NonHomogeneousPoissonProcess)


def get_model_nb_assets(model):
    """
    Gets the number of assets stored by a model (frozen or not).
    """
    from relife.base import FrozenParametricModel
    from relife.lifetime_model import EquilibriumDistribution, MinimumDistribution
    from relife.lifetime_model._regression import LifetimeRegression
    from relife.stochastic_process import NonHomogeneousPoissonProcess, RenewalProcess

    if isinstance(model, EquilibriumDistribution) or isinstance(model, MinimumDistribution):
        return get_model_nb_assets(model.baseline)

    if isinstance(model, NonHomogeneousPoissonProcess):
        return get_model_nb_assets(model.lifetime_model)

    if isinstance(model, RenewalProcess):
        lifetime_model_nb_assets = get_model_nb_assets(model.lifetime_model)
        if model.first_lifetime_model is not None:
            first_lifetime_model_nb_assets = get_model_nb_assets(model.first_lifetime_model)
            return max(lifetime_model_nb_assets, first_lifetime_model_nb_assets)
        return lifetime_model_nb_assets

    if isinstance(model, FrozenParametricModel):
        if isinstance(model._unfrozen_model, NonHomogeneousPoissonProcess):
            return get_model_nb_assets(model._unfrozen_model)
        if isinstance(model._unfrozen_model, LifetimeRegression):
            # specific covar reshape
            reshaped_args = [np.atleast_2d(model.args[0])]
            reshaped_args += [reshape_1d_arg(arg) for arg in model.args[1:]]
        else:
            reshaped_args = [reshape_1d_arg(arg) for arg in model.args]
        return get_args_nb_assets(*reshaped_args)

    return 1
