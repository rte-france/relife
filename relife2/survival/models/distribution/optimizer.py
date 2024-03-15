import numpy as np

from ..interface.distribution import DistOptimizer, ParametricDistLikelihood


class GompertzOptimizer(DistOptimizer):
    def __init__(self, likelihood: ParametricDistLikelihood):
        super().__init__(likelihood)

    def _init_param(self, nb_params: int) -> np.ndarray:
        param0 = np.empty(nb_params)
        rate = np.pi / (
            np.sqrt(6)
            * np.std(
                np.concatenate(
                    [
                        data.values
                        for data in self.likelihood.databook(
                            "complete | right_censored | left_censored"
                        )
                    ]
                )
            )
        )

        c = np.exp(
            -rate
            * np.mean(
                np.concatenate(
                    [
                        data.values
                        for data in self.likelihood.databook(
                            "complete | right_censored | left_censored"
                        )
                    ]
                )
            )
        )

        param0[0] = c
        param0[1] = rate

        return param0
