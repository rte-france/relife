import warnings
from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.core import ParametricModel, Likelihood, LifetimeInterface
from relife2.data import lifetime_factory_template
from relife2.data.dataclass import Sample, LifetimeSample, Truncations
from relife2.io import preprocess_lifetime_data


class LikelihoodFromLifetimes(Likelihood):
    """
    BLABLABLA
    """

    def __init__(
        self,
        model: "ParametricLifetimeModel",
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


class ParametricLifetimeModel(LifetimeInterface, ParametricModel, ABC):
    """
    Extended interface of LifetimeModel whose params can be estimated with fit method
    """

    def fit(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
        args: Optional[Sequence[ArrayLike] | ArrayLike] = (),
        inplace: bool = True,
        **kwargs,
    ) -> np.ndarray:
        """
        BLABLABLABLA
        Args:
            time (ArrayLike):
            event (Optional[ArrayLike]):
            entry (Optional[ArrayLike]):
            departure (Optional[ArrayLike]):
            args (Optional[tuple[ArrayLike]]):
            inplace (bool): (default is True)

        Returns:
            Parameters: optimum parameters found
        """
        time, event, entry, departure, args = preprocess_lifetime_data(
            time, event, entry, departure, args
        )
        observed_lifetimes, truncations = lifetime_factory_template(
            time,
            event,
            entry,
            departure,
            args,
        )

        optimized_function = self.function.copy()
        optimized_function.args = [
            np.empty_like(arg) for arg in args
        ]  # used for init_params if it depends on args
        optimized_function.init_params(observed_lifetimes.rlc)
        param0 = optimized_function.params

        likelihood = LikelihoodFromLifetimes(
            optimized_function,
            observed_lifetimes,
            truncations,
        )

        minimize_kwargs = {
            "method": kwargs.get("method", "L-BFGS-B"),
            "constraints": kwargs.get("constraints", ()),
            "tol": kwargs.get("tol", None),
            "callback": kwargs.get("callback", None),
            "options": kwargs.get("options", None),
            "bounds": kwargs.get("bounds", optimized_function.params_bounds),
            "x0": kwargs.get("x0", param0),
        }

        optimizer = minimize(
            likelihood.negative_log,
            minimize_kwargs.pop("x0"),
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = likelihood.function.params

        return optimizer.x
