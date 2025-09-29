from typing import Generic

from relife import FrozenParametricModel, ParametricModel
from relife._typing import _AdditionalArgs

class StochasticProcess(ParametricModel, Generic[*_AdditionalArgs]): ...

class FrozenStochasticProcess(FrozenParametricModel[*_AdditionalArgs]):
    model: StochasticProcess[*_AdditionalArgs]
    args: tuple[*_AdditionalArgs]
    def __init__(self, model: StochasticProcess[*_AdditionalArgs], *args: *_AdditionalArgs) -> None: ...
