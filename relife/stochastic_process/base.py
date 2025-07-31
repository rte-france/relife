from relife import FrozenParametricModel, ParametricModel


class StochasticProcess(ParametricModel): ...


class FrozenStochasticProcess(FrozenParametricModel):
    def __init__(self, model: StochasticProcess, *args):
        super().__init__(model, *args)
