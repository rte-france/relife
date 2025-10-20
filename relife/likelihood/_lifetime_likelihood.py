import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ._base import Likelihood, FittingResults, approx_hessian
from relife.utils import reshape_1d_arg

def _args_reshape(*args):
    args_list = [np.asarray(arg) for arg in args]
    for i, arg in enumerate(args_list):
        if arg.ndim > 2:
            raise ValueError(
                f"Invalid arg shape, got {arg.shape} shape at position {i}"
            )
        if arg.ndim < 2:
            args_list[i] = arg.reshape(-1, 1)
    return tuple(args_list)


class DefaultLifetimeLikelihood(Likelihood):
    def __init__(self, model, time, *args, event = None, entry = None):
        super().__init__(model)

        time = reshape_1d_arg(time)
        event = reshape_1d_arg(event) if event is not None else np.ones_like(time, dtype=np.bool_)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time, dtype=np.float64)
        args = _args_reshape(*args)
        sizes = [len(x) for x in (time, event, entry, *args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {tuple(sizes)}"
            )

        self._time = time
        self._complete_time = time[np.flatnonzero(event)]
        self._nonzero_entry = entry[np.flatnonzero(entry)]
        self._args = args
        self._complete_time_args = tuple(arg[np.flatnonzero(event)] for arg in args)
        self._nonzero_entry_args = tuple(arg[np.flatnonzero(entry)] for arg in args)

    def _time_contrib(self):
        return np.sum(self.model.chf(self._time, *self._args))

    def _event_contrib(self):
        if len(self._complete_time) == 0:
            return None
        return np.sum(-np.log(self.model.hf(self._complete_time, *self._complete_time_args)))

    def _entry_contrib(self):
        if len(self._nonzero_entry) == 0:
            return None
        return -np.sum(self.model.chf(self._nonzero_entry, *self._nonzero_entry_args))

    def _jac_time_contrib(self):
        jac = self.model.jac_chf(
            self._time,
            *self._args,
            asarray=True,
        )

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def _jac_event_contrib(self):
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

    def _jac_entry_contrib(self):
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

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> np.float64:
        self.params = params  # changes model params
        contributions = (
            self._time_contrib(),
            self._event_contrib(),
            self._entry_contrib(),
        )
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
        self.params = params
        jac_contributions = (
            self._jac_time_contrib(),
            self._jac_event_contrib(),
            self._jac_entry_contrib(),
        )
        return sum(x for x in jac_contributions if x is not None)  # (p,)

    def maximum_likelihood_estimation(self, **optimizer_options) -> FittingResults:
        minimize_kwargs = {
            "method": optimizer_options.get("method", "L-BFGS-B"),
            "constraints": optimizer_options.get("constraints", ()),
            "bounds": optimizer_options.get("bounds", None),
            "x0": optimizer_options.get("x0", self.params),
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
            len(self._time),
            optimal_params,
            neg_log_likelihood,
            covariance_matrix=covariance_matrix,
        )


class IntervalLifetimeLikelihood(Likelihood):
    def __init__(
        self,
        model,
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        *args,
        entry = None,
    ):
        super().__init__(model)
        time_inf = reshape_1d_arg(time_inf)
        time_sup = reshape_1d_arg(time_sup)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time_inf, dtype=np.float64)
        args = _args_reshape(*args)
        sizes = [len(x) for x in (time_inf, time_sup, entry, *args) if x is not None]
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
        self._censored_time_args = tuple(
            arg[~complete_time_index] for arg in args
        )
        self._nonzero_entry_args = tuple(arg[(entry > 0).squeeze()] for arg in args)


    def _complete_time_contrib(self):
        if len(self._complete_time == 0):
            return None
        return np.sum(-np.log(self.model.pdf(self._complete_time, *self._complete_time_args)))

    def _interval_censored_time_contrib(self):
        if len(self._censored_time_upper_bound) == 0:
            return None
        return np.sum(
            -np.log(
                10**-10
                + self.model.cdf(self._censored_time_upper_bound, *self._censored_time_args)
                - self.model.cdf(self._censored_time_lower_bound, *self._censored_time_args)
            ),
        )

    def _entry_contrib(self):
        if len(self._nonzero_entry) == 0:
            return None
        return -np.sum(self.model.chf(self._nonzero_entry, *self._nonzero_entry_args))

    def _jac_complete_time_contrib(self):
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


    def _jac_interval_censored_time_contrib(self):
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


    def _jac_entry_contrib(self):
        if len(self._nonzero_entry) == 0:
            return None
        # filter entry==0 to avoid numerical error in jac_chf
        jac = self.model.jac_chf(
            self._nonzero_entry,
            *self._nonzero_entry_args,
            asarray=True,
        )

        return -np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def negative_log(self, params):
        self.params = params
        contributions = (
            self._complete_time_contrib(),
            self._interval_censored_time_contrib(),
            self._entry_contrib(),
        )
        return sum(x for x in contributions if x is not None)  # ()

    def jac_negative_log(self, params):
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
        return sum(x for x in jac_contributions if x is not None)  # (p,)

    def maximum_likelihood_estimation(self, **optimizer_options):
        minimize_kwargs = {
            "method": optimizer_options.get("method", "L-BFGS-B"),
            "constraints": optimizer_options.get("constraints", ()),
            "bounds": optimizer_options.get("bounds", None),
            "x0": optimizer_options.get("x0", self.model.params),
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
            self._nb_observations,
            optimal_params,
            neg_log_likelihood,
            covariance_matrix=covariance_matrix,
        )
