from typing import overload, TypeVarTuple

from relife.stochastic_process import NonHomogeneousPoissonProcess, FrozenNonHomogeneousPoissonProcess
from relife.lifetime_model import ParametricLifetimeModel, FrozenParametricLifetimeModel

Args = TypeVarTuple("Args")

@overload
def freeze(model : ParametricLifetimeModel[*Args], *args  : *Args) -> FrozenParametricLifetimeModel[*Args]:
    ...

@overload
def freeze(model : NonHomogeneousPoissonProcess[*Args], *args  : *Args) -> FrozenNonHomogeneousPoissonProcess[*Args]:
    ...


def freeze(model: ParametricLifetimeModel[*Args]|NonHomogeneousPoissonProcess[*Args], *args: *Args) -> FrozenParametricLifetimeModel[*Args]|FrozenNonHomogeneousPoissonProcess[*Args]:
    from relife.lifetime_model import LifetimeRegression, LifetimeDistribution

    match model:
        case LifetimeDistribution():
            return model
        case LifetimeRegression():
            from relife.lifetime_model import FrozenLifetimeRegression
            return FrozenLifetimeRegression(model).freeze_args(*args)
        case _ :
            from relife.lifetime_model import FrozenParametricLifetimeModel
            return FrozenParametricLifetimeModel(model).freeze_args(*args)
