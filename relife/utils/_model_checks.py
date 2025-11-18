__all__ = [
    "is_frozen",
    "is_lifetime_model",
    "is_non_homogeneous_poisson_process",
]

from typing import TYPE_CHECKING, Literal, overload

if TYPE_CHECKING:
    from relife.base import FrozenParametricModel, ParametricModel


@overload
def is_frozen(model: FrozenParametricModel[ParametricModel]) -> Literal[True]: ...
@overload
def is_frozen(model: ParametricModel | FrozenParametricModel[ParametricModel]) -> bool: ...
def is_frozen(model: ParametricModel | FrozenParametricModel[ParametricModel]) -> bool:
    """
    Checks if model is frozen
    """
    from relife.base import FrozenParametricModel

    return isinstance(model, FrozenParametricModel)


@overload
def is_lifetime_model(model : ):
def is_lifetime_model(model):
def is_lifetime_model(model):
def is_lifetime_model(model : ParametricModel | ):
    """
    Checks if model is a lifetime model.
    """
    # local import to avoid circular import
    from relife.lifetime_model import ParametricLifetimeModel

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
