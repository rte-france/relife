import copy
from typing import Any, Literal, TypedDict, final, overload

import numpy as np
from numpy._typing import NDArray
from optype.numpy import Array, Array1D, Array2D
from scipy.stats import norm
from typing_extensions import override

from relife.base import FitConfig, FittingResults, MaximumLikelihoodOptimizer
from relife.lifetime_models._parametric_regressions import LinearCovarEffect
from relife.utils import to_column_2d_if_1d

__all__ = [
    "SemiParametricProportionalHazard",
    "CoxPartialLifetimeLikelihood",
    "BreslowPartialLifetimeLikelihood",
    "EfronPartialLifetimeLikelihood",
]


class CoxData:
    time: Array[tuple[int, Literal[1]], np.float64]
    covar: Array2D[np.float64]
    event: Array[tuple[int, Literal[1]], np.bool_] | None
    entry: Array[tuple[int, Literal[1]], np.float64] | None

    ordered_event_time: Array1D[np.float64]
    event_count: Array1D[np.int64]
    risk_set: Array2D[np.bool_]
    death_set: Array2D[np.bool_]
    ordered_event_covar: Array2D[np.float64]

    def __init__(
        self,
        time: Array1D[np.float64],
        covar: Array1D[np.float64] | Array2D[np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
    ) -> None:
        self.time = to_column_2d_if_1d(time)
        self.event = (
            to_column_2d_if_1d(event)
            if event is not None
            else np.ones_like(self.time, dtype=np.bool_)
        )
        self.entry = (
            to_column_2d_if_1d(entry)
            if entry is not None
            else np.zeros_like(self.time, dtype=np.float64)
        )
        self.covar = to_column_2d_if_1d(covar)
        sizes = [len(x) for x in (self.time, self.event, self.entry, self.covar)]

        if len(set(sizes)) != 1:
            raise ValueError(
                f""""
                All lifetime data must have the same number of values. Fields
                length are different. Got {tuple(sizes)}.
                """
            )
        (
            self.ordered_event_time,  # uncensored sorted untied times
            ordered_event_index,
            self.event_count,
        ) = np.unique(
            self.time[self.event == 1],
            return_index=True,
            return_counts=True,
        )
        # here risk_set is mask array on time
        # left truncated & right censored
        self.risk_set = np.logical_and(
            (
                np.vstack([self.entry[:, 0]] * len(self.ordered_event_time))
                < np.hstack([self.ordered_event_time[:, None]] * len(self.time))
            ),
            (
                np.hstack([self.ordered_event_time[:, None]] * len(self.time))
                <= np.vstack([self.time[:, 0]] * len(self.ordered_event_time))
            ),
        )

        self.death_set = np.vstack(
            [self.time[:, 0] * self.event[:, 0]] * len(self.ordered_event_time)
        ) == np.hstack([self.ordered_event_time[:, None]] * len(self.time))

        self.ordered_event_covar = self.covar[self.event[:, 0] == 1][
            ordered_event_index
        ]


def psi(
    covar_effect: LinearCovarEffect,
    data: CoxData,
    on: Literal["risk"] | Literal["death"] = "risk",
    order: Literal[0] | Literal[1] | Literal[2] = 0,
) -> NDArray[np.float64]:
    """Psi formula used for likelihood computations

    Args:
        on (str, optional): "risk" or "death". Defaults to "risk". If "death",
        sum is applied on death set. order (int, optional): order derivatives
        with respect to params. Defaults to 0.

    Returns:
        np.ndarray: psi formulation
        If order 0, shape [m, 1]
        If order 1, shape [m, p]
        If order 2, shape [m, p, p]
    """
    if on == "risk":
        i_set = data.risk_set
    elif on == "death":
        i_set = data.death_set

    if order == 0:
        # shape [m]
        return np.dot(i_set, covar_effect.g(data.covar))
    elif order == 1:
        # shape [m, p]
        return np.dot(i_set, data.covar * covar_effect.g(data.covar))
    elif order == 2:
        # shape [m, p, p]
        return np.tensordot(
            i_set[:, :None],
            data.covar[:, None]
            * data.covar[:, :, None]
            * np.asarray(covar_effect.g(data.covar))[:, :, None],
            axes=1,
        ).astype(np.float64)


class _BreslowBaseline:
    """
    Class for Cox non-parametric Breslow baseline
    """

    data: CoxData
    covar_effect: LinearCovarEffect

    def __init__(self, covar_effect: LinearCovarEffect, data: CoxData):
        assert data.covar.shape[-1] == covar_effect.get_params().size
        self.covar_effect = covar_effect
        self.data = data

    @overload
    def chf(self, se: Literal[False], kp: bool = False) -> NDArray[np.float64]: ...
    @overload
    def chf(
        self, se: Literal[True], kp: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def chf(
        self, se: bool = False, kp: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
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
        """
        # TODO : comprendre le kp
        if kp:
            values = np.cumsum(
                1
                - (
                    1
                    - (
                        self.covar_effect.g(self.data.ordered_event_covar)
                        / psi(self.covar_effect, self.data)
                    )
                )
                ** (self.covar_effect.g(self.data.ordered_event_covar))
            )
        else:
            values = np.cumsum(
                self.data.event_count[:, None] / psi(self.covar_effect, self.data)
            )
        if se:
            var = np.cumsum(
                self.data.event_count[:, None] / psi(self.covar_effect, self.data) ** 2
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
    def sf(self, se: Literal[False]) -> NDArray[np.float64]: ...
    @overload
    def sf(
        self, se: Literal[True]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    def sf(
        self, se: bool = False
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]] | NDArray[np.float64]:
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
        """
        if se:
            chf, chf_conf_int_values = self.chf(se=True)
            return np.exp(-chf), np.exp(-chf_conf_int_values)
        else:
            return np.exp(-self.chf(se=False))


class _SF0(TypedDict):
    timeline: NDArray[np.float64]
    estimation: NDArray[np.float64]


class NotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting."""


class SemiParametricProportionalHazard:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    fitting_results: FittingResults | None
    covar_effect: LinearCovarEffect | None
    _training_data: CoxData | None
    _sf0: _SF0 | None

    def __init__(self):
        self.fitting_results = None
        self.covar_effect = None
        self._sf0 = None
        self._training_data = None

    def get_params(self) -> Array1D[np.float64] | None:
        if self.covar_effect is None:
            return None
        return self.covar_effect.get_params()

    def _require_fitted(
        self,
    ) -> tuple[FittingResults, LinearCovarEffect, CoxData, _SF0]:
        """
        Sort of type narrowing function to check if
        SemiParametricProportionalHazard is fitted.
        """
        if (
            self.fitting_results is None
            or self.covar_effect is None
            or self._training_data is None
            or self._sf0 is None
        ):
            raise NotFittedError("Model not fitted")

        return (
            self.fitting_results,
            self.covar_effect,
            self._training_data,
            self._sf0,
        )

    @overload
    def sf(
        self,
        covar: NDArray[np.float64],
        se: Literal[False],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]: ...
    @overload
    def sf(
        self,
        covar: NDArray[np.float64],
        se: Literal[True],
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]: ...
    @overload
    def sf(
        self, covar: NDArray[np.float64], se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ): ...

    def sf(
        self, covar: NDArray[np.float64], se: bool = True
    ) -> (
        tuple[NDArray[np.float64], NDArray[np.float64]]
        | tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]
    ):
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
        fitting_results, covar_effect, _, sf0 = self._require_fitted()
        if se and fitting_results.covariance_matrix is not None:
            return (
                sf0["timeline"],
                sf0["estimation"] ** covar_effect.g(covar),
                sf0["estimation"] ** covar_effect.g(covar)
                * np.sqrt(self._q1_q2_sum(covar, fitting_results.covariance_matrix)),
            )
        return (
            sf0["timeline"],
            sf0["estimation"] ** covar_effect.g(covar),
        )

    def _q1_q2_sum(
        self, covar: NDArray[np.float64], covariance_matrix: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Klein and Moeschberger: Survival Analysis Techniques for Censored and
        Truncated Data (p. 284).
        """
        _, covar_effect, training_data, _ = self._require_fitted()
        psi_values = psi(covar_effect, training_data)
        psi_order_1 = psi(covar_effect, training_data, order=1)
        d_j_on_psi = training_data.event_count[:, None] / psi_values

        q3 = np.cumsum(
            ((psi_order_1 / psi_values)[None, :, :] - covar[:, None, :])
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

    def init_likelihood(
        self,
        time: Array1D[np.float64],
        covar: Array1D[np.float64] | Array2D[np.float64],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> "CoxPartialLifetimeLikelihood|BreslowPartialLifetimeLikelihood|EfronPartialLifetimeLikelihood":
        # init covar_effect
        covar = to_column_2d_if_1d(covar)
        self.covar_effect = LinearCovarEffect((None,) * covar.shape[-1])

        x0 = kwargs.get("x0", np.random.random(covar.shape[1]))
        config = FitConfig(x0)
        config.scipy_minimize_options["method"] = kwargs.get("method", "trust-exact")
        config.covariance_method = kwargs.get("covariance_method", "exact")

        cox_data = CoxData(time, covar, event=event, entry=entry)
        _, event_count = np.unique(time[event == 1], return_counts=True)
        if (event_count > 3).any():  # efron
            return EfronPartialLifetimeLikelihood(self.covar_effect, cox_data, config)
        if (event_count <= 3).all() and (2 in event_count):  # breslow
            return BreslowPartialLifetimeLikelihood(self.covar_effect, cox_data, config)

        return CoxPartialLifetimeLikelihood(self.covar_effect, cox_data, config)

    def fit(
        self,
        time: Array1D[np.float64],
        covar: Array1D[Any] | Array2D[Any],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ):
        likelihood = self.init_likelihood(time, covar, event, entry, **kwargs)
        self.fitting_results = likelihood.optimize()
        assert self.covar_effect is not None
        self.covar_effect.set_params(self.fitting_results.optimal_params)
        self._training_data = likelihood.data
        # currently only BreslowBaseline is used to compute sf
        baseline = _BreslowBaseline(self.covar_effect, likelihood.data)

        timeline = likelihood.data.ordered_event_time.copy()
        self._sf0 = _SF0(timeline=timeline, estimation=baseline.sf(se=False))
        return self


@final
class CoxPartialLifetimeLikelihood(
    MaximumLikelihoodOptimizer[LinearCovarEffect, CoxData]
):
    data: CoxData
    model: LinearCovarEffect
    config: FitConfig

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log
        if "hess" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["hess"] = self.hess_negative_log

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def negative_log(self, params: Array1D[np.float64]) -> float:
        self.model.set_params(params)
        return -(
            np.log(self.model.g(self.data.ordered_event_covar)).sum()
            - np.log(psi(self.model, self.data)).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            self.data.ordered_event_covar.sum(axis=0)
            - (psi(self.model, self.data, order=1) / psi(self.model, self.data)).sum(
                axis=0
            )
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = psi(self.model, self.data)
        psi_order_1 = psi(self.model, self.data, order=1)

        hessian_part_1 = psi(self.model, self.data, order=2) / psi_order_0[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)


@final
class BreslowPartialLifetimeLikelihood(
    MaximumLikelihoodOptimizer[LinearCovarEffect, CoxData]
):
    data: CoxData
    model: LinearCovarEffect
    config: FitConfig
    s_j: NDArray[np.float64]

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log
        if "hess" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["hess"] = self.hess_negative_log

        self.s_j = np.dot(self.data.death_set, self.data.covar)

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def negative_log(self, params: Array1D[np.float64]) -> float:
        self.model.set_params(params)  # changes model params

        return -(
            np.log(self.model.g(self.s_j)).sum()
            - (
                self.data.event_count[:, None] * np.log(psi(self.model, self.data))
            ).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            self.s_j.sum(axis=0)
            - (
                self.data.event_count[:, None]
                * (psi(self.model, self.data, order=1) / psi(self.model, self.data))
            ).sum(axis=0)
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = psi(self.model, self.data)
        psi_order_1 = psi(self.model, self.data, order=1)

        hessian_part_1 = psi(self.model, self.data, order=2) / psi_order_0[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        return (self.data.event_count[:, None, None] * hessian_part_1).sum(axis=0) - (
            self.data.event_count[:, None, None] * hessian_part_2
        ).sum(axis=0)


@final
class EfronPartialLifetimeLikelihood(
    MaximumLikelihoodOptimizer[LinearCovarEffect, CoxData]
):
    data: CoxData
    model: LinearCovarEffect
    config: FitConfig
    s_j: NDArray[np.float64]
    discount_rates: NDArray[np.float64]
    discount_rates_mask: NDArray[np.bool_]
    scipy_method = "trust-exact"

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ):
        self.model = copy.deepcopy(model)
        self.data = data
        self.config = config
        if "jac" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["jac"] = self.jac_negative_log
        if "hess" not in self.config.scipy_minimize_options:
            self.config.scipy_minimize_options["hess"] = self.hess_negative_log
        self.s_j = np.dot(self.data.death_set, self.data.covar)
        self.discount_rates = (
            np.vstack(
                (np.arange(self.data.event_count.max()),) * len(self.data.event_count)
            )
            / self.data.event_count[:, None]
        )
        self.discount_rates_mask = np.where(self.discount_rates < 1, True, False)

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    def _psi_efron(
        self,
        order: Literal[0] | Literal[1] | Literal[2] = 0,
    ) -> NDArray[np.float64]:
        """Psi formula for Efron method

        Args:
            order (int, optional): order derivatives with respect to params. Defaults to 0.

        Returns:
            np.ndarray: psi formulation for Efron method
            If order 0, shape [m, max(d_j)]
            If order 1, shape [m, max(d_j), p]
            If order 2, shape [m, max(d_j), p, p]
        """  # noqa: E501

        if order == 0:
            # shape [m, max(d_j)]
            return (
                psi(self.model, self.data, order=order) * self.discount_rates_mask
                - psi(self.model, self.data, on="death", order=order)
                * self.discount_rates
                * self.discount_rates_mask
            )
        elif order == 1:
            # shape [m, max(d_j), p]
            return (
                psi(self.model, self.data, order=1)[:, None, :]
                * self.discount_rates_mask[:, :, None]
                - psi(self.model, self.data, on="death", order=1)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None]
            )
        elif order == 2:
            # shape [m, max(d_j), p, p]
            return (
                psi(self.model, self.data, order=2)[:, None, :]
                * self.discount_rates_mask[:, :, None, None]
                - psi(self.model, self.data, on="death", order=2)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None, None]
            )

    @override
    def negative_log(self, params: Array1D[np.float64]) -> float:
        self.model.set_params(params)  # changes model params

        # .sum(axis=1, keepdims=True) --> sum on alpha to d_j
        # .sum() --> sum on j
        # using where in np.log allows to avoid 0. masked elements
        m = self._psi_efron()
        neg_L = -(
            np.log(self.model.g(self.s_j)).sum()
            - np.log(m, out=np.zeros_like(m), where=(m != 0))
            .sum(axis=1, keepdims=True)
            .sum()
        )
        return neg_L

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.set_params(params)  # changes model params
        # .sum(axis=1) --> sum on alpha to d_j
        # .sum(axis=0) --> sum on j
        # using where in np.divide allows to avoid 0. masked elements
        a = self._psi_efron(order=1)
        b = self._psi_efron()[:, :, None]
        return -(
            self.s_j.sum(axis=0)
            - np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
            .sum(axis=1)
            .sum(axis=0)
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = self._psi_efron()
        psi_order_1 = self._psi_efron(order=1)

        # .sum(axis=1) --> sum on alpha to d_j
        # using where in np.divide allows to avoid 0. masked elements
        a = self._psi_efron(order=2)
        b = psi_order_0[:, :, None, None]
        hessian_part_1 = np.divide(a, b, out=np.zeros_like(a), where=(b != 0)).sum(
            axis=1
        )

        # .sum(axis=1) --> sum on alpha to d_j
        # using where in np.divide allows to avoid 0. masked elements
        b = psi_order_0[:, :, None]
        hessian_part_2 = (
            np.divide(psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0))[
                :, :, None, :
            ]
            * (
                np.divide(
                    psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0)
                )
            )[:, :, :, None]
        )
        hessian_part_2 = hessian_part_2.sum(axis=1)

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)
