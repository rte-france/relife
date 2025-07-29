from typing import Generic, TypeVarTuple

from relife import FrozenParametricModel, ParametricModel

Args = TypeVarTuple("Args")


class StochasticProcess(ParametricModel, Generic[*Args]): ...


class FrozenStochasticProcess(FrozenParametricModel[*Args]):
    def __init__(self, model: StochasticProcess[*Args], *args: *Args):
        super().__init__(model, *args)
