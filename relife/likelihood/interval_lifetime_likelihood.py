from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from . import FittingResults, approx_hessian
from ._base import Likelihood

if TYPE_CHECKING:
    from relife.lifetime_model import (
        MinimumDistribution,
    )

    from ..lifetime_model.distribution import LifetimeDistribution
    from ..lifetime_model.regression import LifetimeRegression


def array_reshape(array: NDArray) -> NDArray:
    # Check time shape
    if array.ndim > 2 or (array.ndim == 2 and array.shape[-1] != 1):
        raise ValueError(
            f"Invalid array shape, got {array.shape} be array must be (m,) or (m,1)"
        )
    if array.ndim < 2:
        array = array.reshape(-1, 1)
    return array


def args_reshape(
    args: tuple[float | NDArray[np.float64], ...] = (),
) -> tuple[NDArray[np.float64], ...]:
    args_list: list[NDArray[np.float64]] = [np.asarray(arg) for arg in args]
    for i, arg in enumerate(args_list):
        if arg.ndim > 2:
            raise ValueError(
                f"Invalid arg shape, got {arg.shape} shape at position {i}"
            )
        if arg.ndim < 2:
            args_list[i] = arg.reshape(-1, 1)
    return tuple(args_list)


class IntervalLifetimeLikelihood(Likelihood):
    def __init__(
        self,
        model: Union[LifetimeDistribution, LifetimeRegression, MinimumDistribution],
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        *args,
        entry: NDArray[np.float64] = None,
    ):
        time_inf = array_reshape(time_inf)
        time_sup = array_reshape(time_sup)
        entry = (
            array_reshape(entry)
            if (entry is not None)
            else np.zeros_like(time_inf, dtype=np.float64)
        )
        args = args_reshape(args)

        sizes = [len(x) for x in (time_inf, time_sup, entry, *args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {set(sizes)}"
            )

        data = {"time": time, "event": event, "entry": entry, "args": args}
        super().__init__(model, data)
        self.nb_samples = len(time_inf)

        exact_times_index = (time_inf == time_sup).squeeze()

        self.time_exact = time_sup[exact_times_index]
        self.time_inf = time_inf[~exact_times_index]
        self.time_sup = time_sup[~exact_times_index]

        self.entry = entry[(entry > 0).squeeze()]

        self.args = args
        self.args_with_time_exact = tuple(arg[exact_times_index] for arg in args)
        self.args_with_interval_censoring = tuple(
            arg[~exact_times_index] for arg in args
        )
        self.args_with_entry = tuple(arg[(entry > 0).squeeze()] for arg in args)

    def _interval_censored_contrib(self) -> np.float64:
        if len(self.time_sup) == 0:
            return None
        return np.sum(
            -np.log(
                10**-10
                + self.model.cdf(self.time_sup, *self.args_with_interval_censoring)
                - self.model.cdf(self.time_inf, *self.args_with_interval_censoring)
            ),
            dtype=np.float64,
        )

    def _event_contrib(self) -> np.float64:
        if len(self.time_exact == 0):
            return None
        return -np.sum(
            np.log(self.model.pdf(self.time_exact, *self.args_with_time_exact)),
            dtype=np.float64,
        )

    def _entry_contrib(self) -> np.float64:
        if len(self.entry) == 0:
            return None
        return -np.sum(
            self.model.chf(self.entry, *self.args_with_entry),
            dtype=np.float64,
        )

    def _jac_interval_censored_contrib(self) -> NDArray[np.float64]:
        if len(self.time_sup) == 0:
            return None

        jac_interval_censored = (
            self.model.jac_sf(
                self.time_sup,
                *self.args_with_interval_censoring,
                asarray=True,
            )
            - self.model.jac_sf(
                self.time_inf,
                *self.args_with_interval_censoring,
                asarray=True,
            )
        ) / (
            10**-10
            + self.model.cdf(self.time_sup, *self.args_with_interval_censoring)
            - self.model.cdf(self.time_inf, *self.args_with_interval_censoring)
        )

        return np.sum(
            jac_interval_censored,
            axis=tuple(range(1, jac_interval_censored.ndim)),
            dtype=np.float64,
        )

    def _jac_event_contrib(self) -> NDArray[np.float64]:
        if len(self.time_exact == 0):
            return None
        jac = -self.model.jac_pdf(
            self.time_exact,
            *self.args_with_time_exact,
            asarray=True,
        ) / self.model.pdf(
            self.time_exact,
            *self.args_with_time_exact,
        )

        return np.sum(
            jac,
            axis=tuple(range(1, jac.ndim)),
            dtype=np.float64,
        )

    def _jac_entry_contrib(self) -> NDArray[np.float64]:
        if len(self.entry) == 0:
            return None
        jac = self.model.jac_chf(
            self.entry,  # filter entry==0 to avoid numerical error in jac_chf
            *self.args_with_entry,
            asarray=True,
        )

        return -np.sum(
            jac,
            axis=tuple(range(1, jac.ndim)),
            dtype=np.float64,
        )

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> np.float64:
        model_params = np.copy(self.model.params)
        self.model.params = params  # changes model params
        contributions = (
            self._interval_censored_contrib(),
            self._event_contrib(),
            self._entry_contrib(),
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
            self._jac_interval_censored_contrib(),
            self._jac_event_contrib(),
            self._jac_entry_contrib(),
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
        covariance_matrix = np.linalg.pinv(hessian)
        return FittingResults(
            self.nb_samples,
            optimal_params,
            neg_log_likelihood,
            covariance_matrix=covariance_matrix,
        )
