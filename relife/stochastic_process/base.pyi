from typing import Generic

from relife import FrozenParametricModel, ParametricModel
from relife._typing import _Xs

class StochasticProcess(ParametricModel, Generic[*_Xs]): ...

class FrozenStochasticProcess(FrozenParametricModel[*_Xs]):
    model: StochasticProcess[*_Xs]
    args: tuple[*_Xs]
    def __init__(self, model: StochasticProcess[*_Xs], *args: *_Xs) -> None: ...
