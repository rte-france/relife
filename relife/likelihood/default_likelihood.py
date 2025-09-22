from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from . import FittingResults, approx_hessian

if TYPE_CHECKING:
    from relife.lifetime_model import (
        MinimumDistribution,
    )

    from ..lifetime_model.distribution import LifetimeDistribution
    from ..lifetime_model.regression import LifetimeRegression


class DefaultLikelihood:
    """
    Generic default likelihood object for parametric_model lifetime model

    Parameters
    ----------
    model : ParametricLifetimeDistribution
        Underlying core used to compute probability functions
    time : NDArray
        float, age of the asset at the end of the observation
    event : 
        bool, if the obervsation ends with an event
    entry :
        float, age of the asset at the start of the observation
    """
    def __init__(
        self,
        model: Union[LifetimeDistribution, LifetimeRegression, MinimumDistribution],
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]],
        entry: Optional[NDArray[np.float64]],
    ):
        self.model = model
        self.time = time
        self.event = event
        self.entry = entry

    def _time_contrib(self, time: NDArray[np.float64]) -> np.float64:
        return np.sum(
            self.model.chf(time),
            dtype=np.float64,  # (m, 1)
        )  # ()

    def _event_contrib(
        self, time: NDArray[np.float64], event: NDArray[np.float64]
    ) -> np.float64:
        return -np.sum(
            event
            * np.log(
                self.model.hf(
                    time,
                )
            )
        )  # ()

    def _entry_contrib(self, entry: NDArray[np.float64]) -> np.float64:
        return -np.sum(
            self.model.chf(
                entry,
            ),  # (m, 1)
            dtype=np.float64,
        )  # ()

    def _jac_time_contrib(self, time: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.jac_chf(
            time,
            asarray=True,
        ).sum(axis=1)  # (p,)

    def _jac_event_contrib(
        self, time: NDArray[np.float64], event: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return -np.sum(
            event * (self.model.jac_hf(time, asarray=True) / self.model.hf(time)),
            axis=1,
        )  # (p,)

    def _jac_entry_contrib(self, entry: NDArray[np.float64]) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_chf(entry, asarray=True),
            axis=1,
            dtype=np.float64,
        )

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p+1,)
    ) -> np.float64:
        
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params

        contributions = (
            self._time_contrib(self.time),
            self._event_contrib(self.time, self.event),
            self._entry_contrib(self.entry),
        )
        self.model.params = model_params
        return sum(x for x in contributions if x is not None)

    def jac_negative_log(
        self,
        params: NDArray[np.float64],  # (p+1,)
    ) -> NDArray[np.float64]:
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params
        jac_contributions = (
            self._jac_time_contrib(self.time),
            self._jac_event_contrib(self.time,self.event),
            self._entry_contrib(self.entry)
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
                if minimize_kwargs["method"] not in ("Nelder-Mead", "Powell", "COBYLA", "COBYQA")
                else None
            ),
            **minimize_kwargs,
        )
        optimal_params = np.copy(optimizer.x)
        neg_log_likelihood = np.copy(optimizer.fun)  # neg_log_likelihood value at optimal
        hessian = approx_hessian(self, optimal_params)
        covariance_matrix = np.linalg.inv(hessian)
        return FittingResults(
            len(self.time), optimal_params, neg_log_likelihood, covariance_matrix=covariance_matrix
        )

