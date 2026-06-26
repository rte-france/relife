from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, TypedDict, overload

import numpy as np
from optype.numpy import Array1D, ArrayND
from scipy.stats import norm

from relife.base import FittingResults
from relife.lifetime_models._parametric_regressions import LinearCovarEffect
from relife.likelihoods import init_cox_likelihood
from relife.likelihoods._cox_likelihood import (
    BreslowPartialLifetimeLikelihood,
    CoxPartialLifetimeLikelihood,
    EfronPartialLifetimeLikelihood,
)
from relife.typing import ST, NumpyST
from relife.utils import to_column_2d_if_1d


class CoxEstimation(NamedTuple):
    timeline: Array1D[np.float64]
    values: Array1D[np.float64]
    se: Array1D[np.float64] | None = None


class _SF0(TypedDict):
    timeline: Array1D[np.float64]
    values: Array1D[np.float64]


class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting."""


# TODO : changer le __init__ des non parametric


class SemiParametricProportionalHazard:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    fitting_results: FittingResults
    covar_effect: LinearCovarEffect
    likelihood: (
        CoxPartialLifetimeLikelihood
        | BreslowPartialLifetimeLikelihood
        | EfronPartialLifetimeLikelihood
    )

    def __init__(
        self,
        time: Array1D[np.float64],
        covar: Array1D[np.float64] | Sequence[Array1D[np.float64]],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ):
        nb_covar = 1 if not isinstance(covar, Sequence) else len(covar)
        self.covar_effect = LinearCovarEffect([0.0] * nb_covar)
        self.likelihood = init_cox_likelihood(
            self.covar_effect, time, covar, event, entry, **kwargs
        )
        fitting_results = self.likelihood.optimize()
        self.covar_effect.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results

        timeline = self.likelihood.data.ordered_event_time.copy()
        self._sf0 = _SF0(timeline=timeline, values=self.sf0(se=False))

    def get_params(self) -> Array1D[np.float64]:
        return self.covar_effect.get_params()

    @overload
    def chf0(self, se: Literal[False]) -> ArrayND[np.float64]: ...
    @overload
    def chf0(
        self, se: Literal[True]
    ) -> tuple[ArrayND[np.float64], ArrayND[np.float64]]: ...
    def chf0(
        self, se: bool = False
    ) -> tuple[ArrayND[np.float64], ArrayND[np.float64]] | ArrayND[np.float64]:
        """
        The cumulative hazard function estimation

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline,
            the estimated values and optionally the estimated standard errors (if se is set to true)
        """  # noqa: E501
        values = np.cumsum(
            self.likelihood.data.event_count[:, None] / self.likelihood.psi()
        )
        if se:
            var = np.cumsum(
                self.likelihood.data.event_count[:, None] / self.likelihood.psi() ** 2
            )
            conf_int_values = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )
            return values, conf_int_values
        else:
            return values

    @overload
    def sf0(self, se: Literal[False]) -> ArrayND[np.float64]: ...
    @overload
    def sf0(
        self, se: Literal[True]
    ) -> tuple[ArrayND[np.float64], ArrayND[np.float64]]: ...
    def sf0(
        self, se: bool = False
    ) -> tuple[ArrayND[np.float64], ArrayND[np.float64]] | ArrayND[np.float64]:
        """
        The survival function estimation

        Parameters
        ----------
        se : bool, default is False
            If true, the estimated standard errors are returned too.

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline,
            the estimated values and optionally the estimated standard errors (if se is set to true)
        """  # noqa: E501
        if se:
            chf, chf_conf_int_values = self.chf0(se=True)
            return np.exp(-chf), np.exp(-chf_conf_int_values)
        else:
            return np.exp(-self.chf0(se=False))

    def sf(
        self, *covar: ST | NumpyST | Array1D[NumpyST], se: bool = True
    ) -> CoxEstimation:
        """
        The survival function estimations.

        Parameters
        ----------
        covar: np.array
            array with covariates values
        se : bool, default True
            If True, the standard errors are returned in addition to timeline
            and sf values.

        Returns
        -------
        out : tuple of timeline, values, optionally se. Default is None
            A timeline, corresponding sf values and optionnaly the standard
            errors. If the estimations does not exist yet, returns None.
        """
        if se and self.fitting_results.covariance_matrix is not None:
            return CoxEstimation(
                self._sf0["timeline"],
                self._sf0["values"] ** to_column_2d_if_1d(self.covar_effect.g(*covar)),
                se=self._sf0["values"]
                ** to_column_2d_if_1d(self.covar_effect.g(*covar))
                * np.sqrt(
                    self._q1_q2_sum(
                        *covar, covariance_matrix=self.fitting_results.covariance_matrix
                    )
                ),
            )
        return CoxEstimation(
            self._sf0["timeline"],
            self._sf0["values"] ** to_column_2d_if_1d(self.covar_effect.g(*covar)),
        )

    def _q1_q2_sum(
        self,
        *covar: ST | NumpyST | Array1D[NumpyST],
        covariance_matrix: ArrayND[np.float64],
    ) -> ArrayND[np.float64]:
        """
        Klein and Moeschberger: Survival Analysis Techniques for Censored and
        Truncated Data (p. 284).
        """
        psi_values = self.likelihood.psi()
        psi_order_1 = self.likelihood.psi(order=1)
        d_j_on_psi = self.likelihood.data.event_count[:, None] / psi_values

        q3 = np.cumsum(
            (
                (psi_order_1 / psi_values)[None, :, :]
                - np.column_stack(covar)[:, None, :]
            )
            * d_j_on_psi[None, :, :],
            axis=1,
        )  # [m: new sample for inference, t: timeline, p]
        q2 = np.squeeze(
            np.matmul(
                q3[:, :, None, :],
                np.matmul(
                    covariance_matrix[None, None, :, :],
                    q3[:, :, :, None],
                ),
            )
        )  # [m, t]
        q1 = np.cumsum(d_j_on_psi * (1 / psi_values))
        return q1 + q2
