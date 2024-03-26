import numpy as np

from ..backbone import CovarEffect, ProbabilityFunctions
from ..distributions.functions import DistFunctions


class PHEffect(CovarEffect):
    def __init__(self, nb_covar: int):
        super().__init__(nb_covar)

    def g(self, covar: np.ndarray) -> np.ndarray:
        return np.exp(np.dot(covar, self.params.values[:, None]))

    def log_g(self, covar: np.ndarray) -> np.ndarray:
        return np.dot(covar, self.params.values[:, None])


class PHFunctions(ProbabilityFunctions):
    def __init__(self, nb_covar: int, baseline: DistFunctions):
        self.baseline = baseline()
        self.covar_effect = PHEffect(nb_covar)
        super().__init__(
            self.baseline.params.nb_params
            + self.covar_effect.params.nb_params,
            self.baseline.params.param_names
            + self.covar_effect.params.param_names,
        )

    def hf(self, time: np.ndarray) -> np.ndarray:
        pass

    def chf(self, time: np.ndarray) -> np.ndarray:
        pass
