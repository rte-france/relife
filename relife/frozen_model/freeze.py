from typing import overload, TypeVarTuple, TYPE_CHECKING

if TYPE_CHECKING:
    from relife.lifetime_model import ParametricLifetimeModel, LifetimeRegression
    from .frozen_lifetime_model import FrozenParametricLifetimeModel

Args = TypeVarTuple("Args")

@overload
def freeze(model : ParametricLifetimeModel[*Args], *args  : *Args) -> FrozenParametricLifetimeModel[*Args]:
    ...

def freeze(model: ParametricLifetimeModel[*Args], *args: *Args) -> FrozenParametricLifetimeModel[*Args]:
    from relife.lifetime_model import LifetimeRegression, LifetimeDistribution

    match model:
        case LifetimeDistribution():
            return model
        case LifetimeRegression():
            from .frozen_lifetime_model import FrozenLifetimeRegression
            return FrozenLifetimeRegression(model).freeze_args(*args)
        case _ :
            from .frozen_lifetime_model import FrozenParametricLifetimeModel
            return FrozenParametricLifetimeModel(model).freeze_args(*args)
