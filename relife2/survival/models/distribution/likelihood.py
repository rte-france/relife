import numpy as np

from ...data.base import DataBook
from ..interface.distribution import (
    ParametricDistFunction,
    ParametricDistLikelihood,
)


class ExponentialDistLikelihood(ParametricDistLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)

    # relife/distribution.Exponential
    def jac_hf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1))

    # relife/distribution.Exponential
    def jac_chf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        # shape : (len(sample), nb_param)
        return np.ones((time.size, 1)) * time[:, None]


class WeibullDistLikelihood(ParametricDistLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)

    def jac_hf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:

        return np.column_stack(
            (
                functions.params.rate
                * (functions.params.rate * time[:, None])
                ** (functions.params.c - 1)
                * (
                    1
                    + functions.params.c
                    * np.log(functions.params.rate * time[:, None])
                ),
                functions.params.c**2
                * (functions.params.rate * time[:, None])
                ** (functions.params.c - 1),
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.log(functions.params.rate * time[:, None])
                * (functions.params.rate * time[:, None])
                ** functions.params.c,
                functions.params.c
                * time[:, None]
                * (functions.params.rate * time[:, None])
                ** (functions.params.c - 1),
            )
        )


class GompertzDistLikelihood(ParametricDistLikelihood):
    def __init__(self, databook: DataBook):
        super().__init__(databook)

    def jac_hf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        return np.column_stack(
            (
                functions.params.rate
                * np.exp(functions.params.rate * time[:, None]),
                functions.params.c
                * np.exp(functions.params.rate * time[:, None])
                * (1 + functions.params.rate * time[:, None]),
            )
        )

    def jac_chf(
        self,
        time: np.ndarray,
        functions: ParametricDistFunction,
    ) -> np.ndarray:
        return np.column_stack(
            (
                np.expm1(functions.params.rate * time[:, None]),
                functions.params.c
                * time[:, None]
                * np.exp(functions.params.rate * time[:, None]),
            )
        )
