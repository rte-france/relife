import warnings
from functools import singledispatch
from typing import Union, Any

import numpy as np

from relife2.core import Likelihood, ParametricLifetimeModel, ParametricModel
from relife2.data.dataclass import Sample, LifetimeSample, Truncations


class LikelihoodFromLifetimes(Likelihood):
    """
    BLABLABLA
    """

    def __init__(
        self,
        model: ParametricLifetimeModel,
        observed_lifetimes: LifetimeSample,
        truncations: Truncations,
    ):
        super().__init__(model)
        self.observed_lifetimes = observed_lifetimes
        self.truncations = truncations

    def _complete_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(np.log(self.model.hf(lifetimes.values, *lifetimes.args)))

    def _right_censored_contribs(self, lifetimes: Sample) -> float:
        return np.sum(
            self.model.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _left_censored_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            np.log(-np.expm1(-self.function.chf(lifetimes.values, *lifetimes.args)))
        )

    def _left_truncations_contribs(self, lifetimes: Sample) -> float:
        return -np.sum(
            self.model.chf(lifetimes.values, *lifetimes.args), dtype=np.float64
        )

    def _jac_complete_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.model.jac_hf(lifetimes.values, *lifetimes.args)
            / self.model.hf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_right_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        return np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def _jac_left_censored_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args)
            / np.expm1(self.model.chf(lifetimes.values, *lifetimes.args)),
            axis=0,
        )

    def _jac_left_truncations_contribs(self, lifetimes: Sample) -> np.ndarray:
        return -np.sum(
            self.model.jac_chf(lifetimes.values, *lifetimes.args),
            axis=0,
        )

    def negative_log(
        self,
        params: np.ndarray,
    ) -> float:
        self.params = params
        return (
            self._complete_contribs(self.observed_lifetimes.complete)
            + self._right_censored_contribs(self.observed_lifetimes.rc)
            + self._left_censored_contribs(self.observed_lifetimes.left_censored)
            + self._left_truncations_contribs(self.truncations.left)
        )

    def jac_negative_log(
        self,
        params: np.ndarray,
    ) -> Union[None, np.ndarray]:
        """

        Args:
            params ():

        Returns:

        """
        if not self.hasjac:
            warnings.warn("Functions does not support jac negative likelihood natively")
            return None
        self.params = params
        return (
            self._jac_complete_contribs(self.observed_lifetimes.complete)
            + self._jac_right_censored_contribs(self.observed_lifetimes.rc)
            + self._jac_left_censored_contribs(self.observed_lifetimes.left_censored)
            + self._jac_left_truncations_contribs(self.truncations.left)
        )


@singledispatch
def init_likelihood(model: ParametricModel, *args: Any):
    raise NotImplementedError(f"{model} has no Likelihood")


@init_likelihood.register
def _(model: ParametricLifetimeModel, *args: LifetimeSample | Truncations):
    return LikelihoodFromLifetimes(model, *args)
