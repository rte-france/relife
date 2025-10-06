from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from relife.data import LifetimeData

from . import FittingResults, approx_hessian
from ._base import Likelihood

if TYPE_CHECKING:
    from relife.lifetime_model import (
        MinimumDistribution,
    )

    from ..lifetime_model.distribution import LifetimeDistribution
    from ..lifetime_model.regression import LifetimeRegression


class DefaultLikelihood(Likelihood):
    def __init__(
        self,
        model: Union[LifetimeDistribution, LifetimeRegression, MinimumDistribution],
        time: NDArray[np.float64],
        event: NDArray[np.bool_],
        entry: NDArray[np.float64],
        args=(),
    ):
        sizes = [len(x) for x in (time, event, entry, *args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {set(sizes)}"
            )

        self.model = model
        self.time = time
        self.event = event
        self.entry = entry
        self.args = args
        self.nb_samples = len(time)

    def _time_contrib(self, time: NDArray[np.float64]) -> np.float64:
        return np.sum(
            self.model.chf(time, *self.args),
            dtype=np.float64,
        )

    def _event_contrib(
        self, time: NDArray[np.float64], event: NDArray[np.bool_]
    ) -> np.float64:
        return -np.sum(
            event * np.log(self.model.hf(time, *self.args)),
            dtype=np.float64,
        )

    def _entry_contrib(self, entry: NDArray[np.float64]) -> np.float64:
        return -np.sum(
            self.model.chf(entry, *self.args),
            dtype=np.float64,
        )

    def _jac_time_contrib(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sum(
            self.model.jac_chf(time, *self.args),
            axis=1,
            dtype=np.float64,
        )

    def _jac_event_contrib(
        self, time: NDArray[np.float64], event: NDArray[np.bool_]
    ) -> NDArray[np.float64]:
        return -np.sum(
            event
            * (self.model.jac_hf(time, *self.args) / self.model.hf(time, *self.args)),
            axis=1,
            dtype=np.float64,
        )

    def _jac_entry_contrib(self, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_chf(entry, *self.args),
            axis=1,
            dtype=np.float64,
        )

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> np.float64:
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params
        contributions = (
            self._time_contrib(self.time),
            self._event_contrib(self.time, self.event),
            self._entry_contrib(self.entry),
        )
        self.model.params = model_params  # reset model params (negative_log must not change model params)
        return sum(x for x in contributions if x is not None)  # ()

    def jac_negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> NDArray[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The jacobian (here gradient) is computed with respect to parameters

        Parameters
        ----------
        params : ndarray
            Parameters values on which the jacobian is evaluated

        Returns
        -------
        ndarray
            Jacobian of the negative log likelihood value
        """
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params
        jac_contributions = (
            self._jac_time_contrib(self.time),
            self._jac_event_contrib(self.time, self.event),
            self._jac_entry_contrib(self.entry),
        )
        self.model.params = model_params  # reset model params (jac_negative_log must not change model params)
        return sum(x for x in jac_contributions if x is not None)  # (p,)

    def maximum_likelihood_estimation(self, **options: Any) -> FittingResults:
        # configure and run the optimizer
        minimize_kwargs = {
            "method": options.get("method", "L-BFGS-B"),
            "constraints": options.get("constraints", ()),
            "bounds": options.get("bounds", None),
            "x0": options.get("x0", self.model.params),
        }
        optimizer = minimize(
            self.negative_log,
            minimize_kwargs.pop("x0"),
            jac=(
                self.jac_negative_log
                if minimize_kwargs["method"]
                not in ("Nelder-Mead", "Powell", "COBYLA", "COBYQA")
                else None
            ),
            **minimize_kwargs,
        )
        optimal_params = np.copy(optimizer.x)
        neg_log_likelihood = np.copy(
            optimizer.fun
        )  # neg_log_likelihood value at optimal
        hessian = approx_hessian(self, optimal_params)
        covariance_matrix = np.linalg.inv(hessian)
        return FittingResults(
            self.nb_samples,
            optimal_params,
            neg_log_likelihood,
            covariance_matrix=covariance_matrix,
        )
