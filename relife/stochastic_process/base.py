from typing import TypeVarTuple, Generic

from relife import ParametricModel, FrozenParametricModel

Args = TypeVarTuple("Args")

class StochasticProcess(ParametricModel, Generic[*Args]):
    ...


class FrozenStochasticProcess(FrozenParametricModel[*Args]):
    def __init__(self, model: StochasticProcess[*Args], *args: *Args):
        super().__init__(model, *args)