from typing import overload, Literal

from relife.base import FrozenParametricModel, ParametricModel

class StochasticProcess(ParametricModel): ...

def is_stochastic_process(model: ParametricModel) -> bool: ...
