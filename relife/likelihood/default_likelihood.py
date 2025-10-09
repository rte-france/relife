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


class DefaultLikelihood(Likelihood):
    def __init__(
        self,
        model: Union[LifetimeDistribution, LifetimeRegression, MinimumDistribution],
        time: NDArray[np.float64],
        *args,
        event: NDArray[np.bool_] = None,
        entry: NDArray[np.float64] = None,
    ):
        self.model = model
        self.time = array_reshape(time)
        self.event = (
            array_reshape(event)
            if (event is not None)
            else np.ones_like(self.time).astype(bool)
        )
        self.entry = (
            array_reshape(entry)
            if (entry is not None)
            else np.zeros_like(self.time, dtype=np.float64)
        )

        self.args = args_reshape(args)
        self.nb_samples = len(time)

        sizes = [
            len(x)
            for x in (self.time, self.event, self.entry, *self.args)
            if x is not None
        ]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {set(sizes)}"
            )

        self.time_with_event = self.time[self.event.squeeze()]
        self.entry = self.entry[(self.entry > 0).squeeze()]
        self.event_args = tuple(arg[self.event.squeeze()] for arg in self.args)
        self.entry_args = tuple(arg[(self.entry > 0).squeeze()] for arg in self.args)

    def _time_contrib(self, time: NDArray[np.float64], *args) -> np.float64:
        return np.sum(
            self.model.chf(time, *args),
            dtype=np.float64,
        )

    def _event_contrib(
        self, time_with_event: NDArray[np.float64], *event_args
    ) -> np.float64:
        if len(time_with_event) == 0:
            return None
        return -np.sum(
            np.log(self.model.hf(time_with_event, *event_args)),
            dtype=np.float64,
        )

    def _entry_contrib(self, entry: NDArray[np.float64], *entry_args) -> np.float64:
        if len(entry) == 0:
            return None
        return -np.sum(
            self.model.chf(entry, *entry_args),
            dtype=np.float64,
        )

    def _jac_time_contrib(
        self, time: NDArray[np.float64], *args
    ) -> NDArray[np.float64]:
        jac = self.model.jac_chf(
            time,
            *args,
            asarray=True,
        )

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(
            jac,
            axis=tuple(range(1, jac.ndim)),
            dtype=np.float64,
        )

    def _jac_event_contrib(
        self, time_with_event: NDArray[np.float64], *event_args
    ) -> NDArray[np.float64]:
        if len(time_with_event) == 0:
            return None
        jac = -self.model.jac_hf(
            time_with_event,
            *event_args,
            asarray=True,
        ) / self.model.hf(time_with_event, *event_args)

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(
            jac,
            axis=tuple(range(1, jac.ndim)),
            dtype=np.float64,
        )

    def _jac_entry_contrib(
        self, entry: NDArray[np.float64], *entry_args
    ) -> NDArray[np.float64]:
        if len(entry) == 0:
            return None

        jac = -self.model.jac_chf(
            entry,  # filter entry==0 to avoid numerical error in jac_chf
            *entry_args,
            asarray=True,
        )

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(
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
            self._time_contrib(self.time, *self.args),
            self._event_contrib(self.time_with_event, *self.event_args),
            self._entry_contrib(self.entry, *self.entry_args),
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
            self._jac_time_contrib(self.time, *self.args),
            self._jac_event_contrib(self.time_with_event, *self.event_args),
            self._jac_entry_contrib(self.entry, *self.entry_args),
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
