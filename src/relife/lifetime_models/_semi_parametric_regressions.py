from abc import ABC
from collections.abc import Sequence
from typing import Any, Literal, NamedTuple, final

import numpy as np
from optype.numpy import Array, Array1D, Array2D, ArrayND
from typing_extensions import override

from relife.base import FitConfig, FittingResults, MaximumLikelihoodOptimizer
from relife.typing import ST, NumpyST
from relife.utils import to_column_2d_if_1d

from ._parametric_regressions import LinearCovarEffect


class CoxData:
    time: Array[tuple[int, Literal[1]], np.float64]
    covar: tuple[Array[tuple[int, Literal[1]], np.float64], ...]
    event: Array[tuple[int, Literal[1]], np.bool_] | None
    entry: Array[tuple[int, Literal[1]], np.float64] | None

    ordered_event_time: Array1D[np.float64]
    event_count: Array1D[np.int64]
    risk_set: Array2D[np.bool_]
    death_set: Array2D[np.bool_]
    ordered_event_covar: tuple[Array[tuple[int, Literal[1]], np.float64], ...]

    def __init__(
        self,
        time: Array1D[np.float64],
        covar: Sequence[Array1D[np.float64]],
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
        self.covar = tuple(to_column_2d_if_1d(c) for c in covar)
        sizes = [len(x) for x in (self.time, self.event, self.entry, *self.covar)]

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

        self.ordered_event_covar = tuple(
            c[self.event[:, 0] == 1][ordered_event_index] for c in self.covar
        )


class CoxPartialLifetimeLikelihood(
    MaximumLikelihoodOptimizer[LinearCovarEffect, CoxData], ABC
):
    model: LinearCovarEffect
    data: CoxData
    config: FitConfig

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ):
        self.model = model
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
            np.log(self.model.g(*self.data.ordered_event_covar)).sum()
            - np.log(self.psi()).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            np.column_stack(self.data.ordered_event_covar).sum(axis=0)
            - (self.psi(order=1) / self.psi()).sum(axis=0)
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = self.psi()
        psi_order_1 = self.psi(order=1)

        hessian_part_1 = self.psi(order=2) / psi_order_0[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)

    def psi(
        self,
        on: Literal["risk"] | Literal["death"] = "risk",
        order: Literal[0] | Literal[1] | Literal[2] = 0,
    ) -> ArrayND[np.float64]:
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
            i_set = self.data.risk_set
        elif on == "death":
            i_set = self.data.death_set

        if order == 0:
            # shape [m]
            return np.dot(i_set, self.model.g(*self.data.covar))
        elif order == 1:
            # shape [m, p]
            return np.dot(
                i_set,
                np.column_stack(self.data.covar) * self.model.g(*self.data.covar),
            )
        elif order == 2:
            # shape [m, p, p]
            return np.tensordot(
                i_set[:, :None],
                np.column_stack(self.data.covar)[:, None]
                * np.column_stack(self.data.covar)[:, :, None]
                * np.asarray(self.model.g(*self.data.covar))[:, :, None],
                axes=1,
            ).astype(np.float64)


@final
class BreslowPartialLifetimeLikelihood(CoxPartialLifetimeLikelihood):
    s_j: ArrayND[np.float64]

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ):
        super().__init__(model, data, config)

        self.s_j = np.dot(self.data.death_set, np.column_stack(self.data.covar))

    @override
    def negative_log(self, params: Array1D[np.float64]) -> float:
        self.model.set_params(params)  # changes model params

        return -(
            np.log(self.model.g(*np.unstack(self.s_j, axis=-1))).sum()
            - (self.data.event_count[:, None] * np.log(self.psi())).sum()
        )

    @override
    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.set_params(params)  # changes model params

        return -(
            self.s_j.sum(axis=0)
            - (self.data.event_count[:, None] * (self.psi(order=1) / self.psi())).sum(
                axis=0
            )
        )

    @override
    def hess_negative_log(self, params: Array1D[np.float64]) -> Array2D[np.float64]:
        self.model.set_params(params)  # changes model params

        psi_order_0 = self.psi()
        psi_order_1 = self.psi(order=1)

        hessian_part_1 = self.psi(order=2) / psi_order_0[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
            psi_order_1 / psi_order_0
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        return (self.data.event_count[:, None, None] * hessian_part_1).sum(axis=0) - (
            self.data.event_count[:, None, None] * hessian_part_2
        ).sum(axis=0)


@final
class EfronPartialLifetimeLikelihood(CoxPartialLifetimeLikelihood):
    s_j: ArrayND[np.float64]
    discount_rates: ArrayND[np.float64]
    discount_rates_mask: ArrayND[np.bool_]
    scipy_method = "trust-exact"

    def __init__(
        self,
        model: LinearCovarEffect,
        data: CoxData,
        config: FitConfig,
    ):
        super().__init__(model, data, config)
        self.s_j = np.dot(self.data.death_set, np.column_stack(self.data.covar))
        self.discount_rates = (
            np.vstack(
                [np.arange(np.max(self.data.event_count))] * len(self.data.event_count)
            )
            / self.data.event_count[:, None]
        )
        self.discount_rates_mask = np.where(self.discount_rates < 1, True, False)

    def _psi_efron(
        self,
        order: Literal[0] | Literal[1] | Literal[2] = 0,
    ) -> ArrayND[np.float64]:
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
                self.psi(order=order) * self.discount_rates_mask
                - self.psi(on="death", order=order)
                * self.discount_rates
                * self.discount_rates_mask
            )
        elif order == 1:
            # shape [m, max(d_j), p]
            return (
                self.psi(order=1)[:, None, :] * self.discount_rates_mask[:, :, None]
                - self.psi(on="death", order=1)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None]
            )
        elif order == 2:
            # shape [m, max(d_j), p, p]
            return (
                self.psi(order=2)[:, None, :]
                * self.discount_rates_mask[:, :, None, None]
                - self.psi(on="death", order=2)[:, None, :]
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
            np.log(self.model.g(*np.unstack(self.s_j, axis=-1))).sum()
            - np.log(m, out=np.zeros_like(m), where=(m != 0))
            .sum(axis=1, keepdims=True)
            .sum()
        )
        return neg_L

    @override
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

    @override
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


class CoxEstimation(NamedTuple):
    timeline: Array1D[np.float64]
    values: Array1D[np.float64]
    se: Array1D[np.float64] | None = None


class SemiParametricProportionalHazard:
    """
    Class for Cox, semi-parametric, Proportional Hazards, model
    """

    fitting_results: FittingResults
    _covar_effect: LinearCovarEffect
    _likelihood: CoxPartialLifetimeLikelihood
    _sf0_estimation: CoxEstimation

    def __init__(
        self,
        time: Array1D[np.float64],
        covar: Array1D[np.float64] | Sequence[Array1D[np.float64]],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ):
        nb_covar = 1 if not isinstance(covar, Sequence) else len(covar)
        self._covar_effect = LinearCovarEffect([0.0] * nb_covar)
        covar = (covar,) if not isinstance(covar, Sequence) else covar
        self._likelihood = self.init_likelihood(time, covar, event, entry, **kwargs)
        fitting_results = self._likelihood.optimize()
        self._covar_effect.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        timeline = self._likelihood.data.ordered_event_time.copy()
        self._sf0_estimation = CoxEstimation(timeline=timeline, values=self._sf0())

    def init_likelihood(
        self,
        time: Array1D[np.float64],
        covar: Sequence[Array1D[np.float64]],
        event: Array1D[np.bool_] | None = None,
        entry: Array1D[np.float64] | None = None,
        **kwargs: Any,
    ) -> CoxPartialLifetimeLikelihood:
        covar_effect = LinearCovarEffect([0.0] * len(covar))
        x0 = kwargs.get("x0", np.random.random(len(covar)))
        config = FitConfig(x0)
        config.scipy_minimize_options["method"] = kwargs.get("method", "trust-exact")
        config.covariance_method = kwargs.get("covariance_method", "exact")
        cox_data = CoxData(time, covar, event=event, entry=entry)
        _, event_count = np.unique(time[event == 1], return_counts=True)
        if (event_count > 3).any():  # efron
            return EfronPartialLifetimeLikelihood(self._covar_effect, cox_data, config)
        if (event_count <= 3).all() and (2 in event_count):
            return BreslowPartialLifetimeLikelihood(covar_effect, cox_data, config)
        return CoxPartialLifetimeLikelihood(covar_effect, cox_data, config)

    def get_params(self) -> Array1D[np.float64]:
        return self._covar_effect.get_params()

    def _chf0(self) -> ArrayND[np.float64]:
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
        return np.cumsum(
            self._likelihood.data.event_count[:, None] / self._likelihood.psi()
        )
        # if se:
        #     var = np.cumsum(
        #         self._likelihood.data.event_count[:, None] / self._likelihood.psi() ** 2  # noqa: E501
        #     )
        #     conf_int_values = np.hstack(
        #         [
        #             values[:, None]
        #             + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
        #             values[:, None]
        #             - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
        #         ]
        #     )
        #     return values, conf_int_values

    def _sf0(self) -> ArrayND[np.float64]:
        """
        The survival function estimation

        Returns
        -------
        tuple of 2 or 3 ndarrays
            A tuple containing the timeline,
            the estimated values and optionally the estimated standard errors (if se is set to true)
        """  # noqa: E501
        return np.exp(-self._chf0())
        # if se:
        #     chf, chf_conf_int_values = self.chf0(se=True)
        #     return np.exp(-chf), np.exp(-chf_conf_int_values)

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
                self._sf0_estimation.timeline,
                self._sf0_estimation.values
                ** to_column_2d_if_1d(self._covar_effect.g(*covar)),
                se=self._sf0_estimation.values
                ** to_column_2d_if_1d(self._covar_effect.g(*covar))
                * np.sqrt(
                    self._q1_q2_sum(
                        *covar, covariance_matrix=self.fitting_results.covariance_matrix
                    )
                ),
            )
        return CoxEstimation(
            self._sf0_estimation.timeline,
            self._sf0_estimation.values
            ** to_column_2d_if_1d(self._covar_effect.g(*covar)),
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
        psi_values = self._likelihood.psi()
        psi_order_1 = self._likelihood.psi(order=1)
        d_j_on_psi = self._likelihood.data.event_count[:, None] / psi_values

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
