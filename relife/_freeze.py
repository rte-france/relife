from typing import TypeVarTuple, overload

from relife import FrozenParametricModel, ParametricModel
from relife.lifetime_model import (
    FrozenLifetimeRegression,
    FrozenParametricLifetimeModel,
    LifetimeRegression,
    ParametricLifetimeModel,
)
from relife.stochastic_process import (
    FrozenNonHomogeneousPoissonProcess,
    NonHomogeneousPoissonProcess,
)

Args = TypeVarTuple("Args")


@overload
def freeze(model: ParametricModel, *args: *Args) -> FrozenParametricModel[*Args]: ...


@overload
def freeze(model: ParametricLifetimeModel[*Args], *args: *Args) -> FrozenParametricLifetimeModel[*Args]: ...


@overload
def freeze(model: LifetimeRegression[*Args], *args: *Args) -> FrozenLifetimeRegression[*Args]: ...


@overload
def freeze(model: NonHomogeneousPoissonProcess[*Args], *args: *Args) -> FrozenNonHomogeneousPoissonProcess[*Args]: ...


def freeze(
    model: (
        ParametricModel
        | ParametricLifetimeModel[*Args]
        | LifetimeRegression[*Args]
        | NonHomogeneousPoissonProcess[*Args]
    ),
    *args: *Args,
) -> (
    FrozenParametricModel[*Args]
    | FrozenParametricLifetimeModel[*Args]
    | FrozenLifetimeRegression[*Args]
    | FrozenNonHomogeneousPoissonProcess[*Args]
):
    from relife.lifetime_model import (
        FrozenParametricLifetimeModel,
        LifetimeDistribution,
        LifetimeRegression,
    )

    match model:
        case ParametricModel():
            return FrozenParametricModel(model, *args)
        case ParametricLifetimeModel():
            return FrozenParametricLifetimeModel(model, *args)
        case LifetimeRegression():
            return FrozenLifetimeRegression(model, *args)
        case LifetimeDistribution():
            raise ValueError("LifetimeDistribution does not need to be frozen")
        case _:
            raise ValueError(f"{type(model)} can't be be frozen")


def isfrozen(model: ParametricLifetimeModel[*Args] | NonHomogeneousPoissonProcess[*Args]) -> bool:
    if isinstance(model, FrozenParametricLifetimeModel):
        return True
    if isinstance(model, FrozenNonHomogeneousPoissonProcess):
        return True
    return False


@overload
def get_frozen_args(model: ParametricLifetimeModel[*Args] | NonHomogeneousPoissonProcess[*Args]) -> tuple[()]: ...


@overload
def get_frozen_args(
    model: FrozenParametricLifetimeModel[*Args] | FrozenNonHomogeneousPoissonProcess[*Args],
) -> tuple[*Args]: ...


def get_frozen_args(
    model: (
        ParametricLifetimeModel[*Args]
        | NonHomogeneousPoissonProcess[*Args]
        | FrozenParametricLifetimeModel[*Args]
        | FrozenNonHomogeneousPoissonProcess[*Args]
    ),
) -> tuple[*Args] | tuple[()]:
    return model.getattr(model, "frozen_args", ())
