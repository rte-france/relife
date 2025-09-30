from relife.base import ParametricModel

class StochasticProcess(ParametricModel): ...

def is_stochastic_process(model):
    return isinstance(model, StochasticProcess)
