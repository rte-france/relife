from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, Unpack, final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, minimize
from typing_extensions import override

from relife.lifetime_model._base import FittableParametricLifetimeModel
from relife.utils import reshape_1d_arg

from ._base import DifferentiableLikelihood, FittingResults, approx_hessian

if TYPE_CHECKING:
    from relife.typing import ScipyMinimizeOptions

__all__ = ["DefaultLifetimeLikelihood", "IntervalLifetimeLikelihood"]


@final
class DefaultLifetimeLikelihood(DifferentiableLikelihood):

    _nb_observations: int
    _time: NDArray[np.float64]
    _complete_time: NDArray[np.float64]
    _nonzero_entry: NDArray[np.float64]
    _args: tuple[NDArray[Any], ...]
    _complete_time_args: tuple[NDArray[Any], ...]
    _nonzero_entry_args: tuple[NDArray[Any], ...]

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[Any, ...]],
        time: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        event: NDArray[np.bool_] | None = None,
        entry: NDArray[np.float64] | None = None,
    ):
        super().__init__(model)
        time = reshape_1d_arg(time)
        event = reshape_1d_arg(event) if event is not None else np.ones_like(time, dtype=np.bool_)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time, dtype=np.float64)
        if isinstance(model_args, tuple):
            args = tuple((reshape_1d_arg(arg) for arg in model_args))
        elif isinstance(model_args, np.ndarray):
            args = (reshape_1d_arg(model_args),)
        elif model_args is None:
            args = ()
        sizes = [len(x) for x in (time, event, entry, *args)]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {tuple(sizes)}"
            )

        self._time = time
        self._nb_observations = len(time)
        self._complete_time = time[np.flatnonzero(event)]
        self._nonzero_entry = entry[np.flatnonzero(entry)]
        self._args = args
        self._complete_time_args = tuple(arg[np.flatnonzero(event)] for arg in args)
        self._nonzero_entry_args = tuple(arg[np.flatnonzero(entry)] for arg in args)

    def _time_contrib(self) -> np.float64:
        return np.sum(self.model.chf(self._time, *self._args))

    def _event_contrib(self) -> np.float64 | None:
        if len(self._complete_time) == 0:
            return None
        return np.sum(-np.log(self.model.hf(self._complete_time, *self._complete_time_args)))

    def _entry_contrib(self) -> np.float64 | None:
        if len(self._nonzero_entry) == 0:
            return None
        return -np.sum(self.model.chf(self._nonzero_entry, *self._nonzero_entry_args))

    def _jac_time_contrib(self) -> NDArray[np.float64]:
        jac = self.model.jac_chf(
            self._time,
            *self._args,
            asarray=True,
        )

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def _jac_event_contrib(self) -> NDArray[np.float64] | None:
        if len(self._complete_time) == 0:
            return None
        jac = -self.model.jac_hf(
            self._complete_time,
            *self._complete_time_args,
            asarray=True,
        ) / self.model.hf(self._complete_time, *self._complete_time_args)

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def _jac_entry_contrib(self) -> NDArray[np.float64] | None:
        if len(self._nonzero_entry) == 0:
            return None

        # filter entry==0 to avoid numerical error in jac_chf
        jac = -self.model.jac_chf(
            self._nonzero_entry,
            *self._nonzero_entry_args,
            asarray=True,
        )

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(jac, axis=tuple(range(1, jac.ndim)))

    @override
    def negative_log(
        self,
        params: NDArray[np.float64],
    ) -> float:
        self.params = params  # changes model params
        contributions = (
            self._time_contrib(),
            self._event_contrib(),
            self._entry_contrib(),
        )
        return sum(x for x in contributions if x is not None)  # ()

    @override
    def jac_negative_log(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Jacobian of the negative log likelihood.

        The jacobian (here gradient) is computed with respect to parameters

        Parameters
        ----------
        params : ndarray Parameters values on which the jacobian is evaluated Returns
        -------
        ndarray
            Jacobian of the negative log likelihood value
        """
        self.params = params
        jac_contributions = (
            self._jac_time_contrib(),
            self._jac_event_contrib(),
            self._jac_entry_contrib(),
        )
        return np.asarray(sum(x for x in jac_contributions if x is not None))  # (p,)

    @override
    def maximum_likelihood_estimation(self, **optimizer_options: Unpack[ScipyMinimizeOptions]) -> FittingResults:
        x0: NDArray[np.float64] = optimizer_options.get("x0", self.params)
        method: str = optimizer_options.get("method", "L-BFGS-B")
        bounds: Bounds | None = optimizer_options.get("bounds", None)
        if method in ("Nelder-Mead", "Powell", "COBYLA", "COBYQA"):
            optimizer = minimize(
                self.negative_log,
                x0,
                bounds=bounds,
            )
        else:
            optimizer = minimize(
                self.negative_log,
                x0,
                jac=self.jac_negative_log,
                bounds=bounds,
            )
        optimal_params = np.copy(optimizer.x)
        optimal_neg_log_likelihood = copy(optimizer.fun)
        hessian = approx_hessian(self, optimal_params)
        covariance_matrix = np.linalg.pinv(hessian).astype(np.float64)
        return FittingResults(
            self._nb_observations,
            optimal_params,
            optimal_neg_log_likelihood,
            covariance_matrix=covariance_matrix,
        )


@final
class IntervalLifetimeLikelihood(DifferentiableLikelihood):
    _nb_observations: int
    _complete_time: NDArray[np.float64]
    _censored_time_lower_bound: NDArray[np.float64]
    _censored_time_upper_bound: NDArray[np.float64]
    _nonzero_entry: NDArray[np.float64]
    _complete_time_args: tuple[NDArray[Any], ...]
    _censored_time_args: tuple[NDArray[Any], ...]
    _nonzero_entry_args: tuple[NDArray[Any], ...]

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[Any, ...]],
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        entry: NDArray[np.float64] | None = None,
        params0: NDArray[np.float64] | None = None,
    ):
        super().__init__(model)
        time_inf = reshape_1d_arg(time_inf)
        time_sup = reshape_1d_arg(time_sup)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time_inf, dtype=np.float64)
        if isinstance(model_args, tuple):
            args = tuple((reshape_1d_arg(arg) for arg in model_args))
        elif isinstance(model_args, np.ndarray):
            args = (reshape_1d_arg(model_args),)
        elif model_args is None:
            args = ()

        sizes = [len(x) for x in (time_inf, time_sup, entry, *args)]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {tuple(sizes)}"
            )

        self._nb_observations = len(time_inf)

        complete_time_index = np.flatnonzero(time_inf == time_sup)
        self._complete_time = time_sup[complete_time_index]
        self._censored_time_lower_bound = time_inf[~complete_time_index]
        self._censored_time_upper_bound = time_sup[~complete_time_index]

        self._nonzero_entry = entry[(entry > 0).squeeze()]

        self._complete_time_args = tuple(arg[complete_time_index] for arg in args)
        self._censored_time_args = tuple(arg[~complete_time_index] for arg in args)
        self._nonzero_entry_args = tuple(arg[(entry > 0).squeeze()] for arg in args)

    def _complete_time_contrib(self) -> np.float64 | None:
        if len(self._complete_time == 0):
            return None
        return np.sum(-np.log(self.model.pdf(self._complete_time, *self._complete_time_args)))

    def _interval_censored_time_contrib(self) -> np.float64 | None:
        if len(self._censored_time_upper_bound) == 0:
            return None
        return np.sum(
            -np.log(
                10**-10
                + self.model.cdf(self._censored_time_upper_bound, *self._censored_time_args)
                - self.model.cdf(self._censored_time_lower_bound, *self._censored_time_args)
            ),
        )

    def _entry_contrib(self) -> np.float64 | None:
        if len(self._nonzero_entry) == 0:
            return None
        return -np.sum(self.model.chf(self._nonzero_entry, *self._nonzero_entry_args))

    def _jac_complete_time_contrib(self) -> NDArray[np.float64] | None:
        if len(self._complete_time == 0):
            return None
        jac = -self.model.jac_pdf(
            self._complete_time,
            *self._complete_time_args,
            asarray=True,
        ) / self.model.pdf(
            self._complete_time,
            *self._complete_time_args,
        )

        return np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def _jac_interval_censored_time_contrib(self) -> NDArray[np.float64] | None:
        if len(self._censored_time_upper_bound) == 0:
            return None

        jac_interval_censored = (
            self.model.jac_sf(
                self._censored_time_upper_bound,
                *self._censored_time_args,
                asarray=True,
            )
            - self.model.jac_sf(
                self._censored_time_lower_bound,
                *self._censored_time_args,
                asarray=True,
            )
        ) / (
            10**-10
            + self.model.cdf(self._censored_time_upper_bound, *self._censored_time_args)
            - self.model.cdf(self._censored_time_lower_bound, *self._censored_time_args)
        )

        return np.sum(jac_interval_censored, axis=tuple(range(1, jac_interval_censored.ndim)))

    def _jac_entry_contrib(self) -> NDArray[np.float64] | None:
        if len(self._nonzero_entry) == 0:
            return None
        # filter entry==0 to avoid numerical error in jac_chf
        jac = self.model.jac_chf(
            self._nonzero_entry,
            *self._nonzero_entry_args,
            asarray=True,
        )

        return -np.sum(jac, axis=tuple(range(1, jac.ndim)))

    @override
    def negative_log(self, params: NDArray[np.float64]) -> float:
        self.params = params
        contributions = (
            self._complete_time_contrib(),
            self._interval_censored_time_contrib(),
            self._entry_contrib(),
        )
        return sum(x for x in contributions if x is not None)  # ()

    @override
    def jac_negative_log(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
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
        self.params = params
        jac_contributions = (
            self._jac_interval_censored_time_contrib(),
            self._jac_complete_time_contrib(),
            self._jac_entry_contrib(),
        )
        return np.asarray(sum(x for x in jac_contributions if x is not None))  # (p,)

    @override
    def maximum_likelihood_estimation(self, **optimizer_options: Unpack[ScipyMinimizeOptions]) -> FittingResults:
        x0: NDArray[np.float64] = optimizer_options.get("x0", self.params)
        method: str = optimizer_options.get("method", "L-BFGS-B")
        bounds: Bounds | None = optimizer_options.get("bounds", None)
        if method in ("Nelder-Mead", "Powell", "COBYLA", "COBYQA"):
            optimizer = minimize(
                self.negative_log,
                x0,
                bounds=bounds,
            )
        else:
            optimizer = minimize(
                self.negative_log,
                x0,
                jac=self.jac_negative_log,
                bounds=bounds,
            )
        optimal_params: NDArray[np.float64] = np.copy(optimizer.x)
        optimal_neg_log_likelihood: float = copy(optimizer.fun)
        hessian = approx_hessian(self, optimal_params)
        covariance_matrix = np.linalg.pinv(hessian).astype(np.float64)
        return FittingResults(
            self._nb_observations,
            optimal_params,
            optimal_neg_log_likelihood,
            covariance_matrix=covariance_matrix,
        )
