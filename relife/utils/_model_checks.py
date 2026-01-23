from __future__ import annotations

__all__ = [
    "is_frozen",
    "is_lifetime_model",
    "is_non_homogeneous_poisson_process",
]

from typing import TYPE_CHECKING, Any, Literal, TypeVarTuple, overload

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
