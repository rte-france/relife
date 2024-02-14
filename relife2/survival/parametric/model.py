from .. import SurvivalData
from .function import ExponentialDistriFunction, ParametricDistriFunction
from .likelihood import ExponentialDistriLikelihood, ParametricDistriLikelihood
from .optimizer import DistriOptimizer


class ParametricDistriModel:
    def __init__(
        self,
        data: SurvivalData,
        functions: ParametricDistriFunction,
        likelihood: ParametricDistriLikelihood,
        # optimizer: DistriOptimizer,
    ):

        assert isinstance(data, SurvivalData)
        assert issubclass(functions, ParametricDistriFunction)
        assert issubclass(likelihood, ParametricDistriLikelihood)
        # assert issubclass(optimizer, DistriOptimizer)
        self.data = data
        self.functions = functions
        self.likelihood = likelihood
        self.optimizer = DistriOptimizer()

    def sf():
        pass

    def fit():
        pass


def exponential(data: SurvivalData):
    return ParametricDistriModel(
        data, ExponentialDistriFunction, ExponentialDistriLikelihood
    )


def gompertz(data: SurvivalData):
    pass
