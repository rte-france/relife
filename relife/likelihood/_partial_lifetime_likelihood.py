import copy
from typing import Literal, final

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D, ToFloat, ToFloat2D
from scipy.optimize import Bounds
from typing_extensions import override

from relife.lifetime_model._regression import LinearCovarEffect
from relife.lifetime_model._semi_parametric import CoxData, psi

from ._base import MaximumLikehoodOptimizer

__all__ = [
    "CoxPartialLifetimeLikelihood",
    "BreslowPartialLifetimeLikelihood",
    "EfronPartialLifetimeLikelihood",
]


r"""
Manually specify method used to compute partial likelihood and its
derivates. By default :code:`method` is set automatically.

Args:
    method (str): "cox", "breslow" or "efron"

Notes:
    Based on :code:`method` value, the computation of the partial negative log partial likelihood differs :
        - If "cox", :math:`\sum_j^{m}\ln\left( \psi_{\mathbf{R}_j}(\vec{z}_i)\right) - \sum_j^{m}\ln(g(\vec{z}_{(j)}))`
        - If "breslow", :math:`\sum_j^{m} d_j \ln\left( \psi\right) - \sum_j^{m}\ln(g(\vec{s}_j))`
        - If "efron", :math:`\sum_j^{m} \sum_{\alpha}^{d_j} \ln\left( \psi_{\mathbf{R}_j} - \frac{\alpha -1}{d_j} \psi_{\mathbf{D}_j}\right) - \sum_j^{m}\ln(g(\vec{s}_j))`

    When Cox or Breslow methods are used, psi is defined as followed :
        - Order 0 derivative, :math:`\psi_{\mathbf{R}_j}(\vec{z}_i) = \sum_{i\in\mathbf{R}(v_j)}g(\vec{z}_i)`
        - Order 1 derivative, :math:`\psi_{\mathbf{R}_j}(k, \vec{z}_i) = \sum_{i\in\mathbf{R}(v_j)} z_{ik} \cdot g(\vec{z}_i)`
        - Order 2 derivative, :math:`\psi_{\mathbf{R}_j}(h,~k,~\vec{z}_i) = \sum_{i\in\mathbf{R}(v_j)} z_{ik} \cdot z_{ih} \cdot g(\vec{z}_i)`
    See :code:`Cox(*args)._psi(*args)`

    When Efron method is used, psi is defined as followed :
        - Order 0 derivative, :math:`\psi_{\mathbf{R}_j}(\vec{z}_i) = \sum_{i\in\mathbf{R}(v_j)}g(\vec{z}_i)`
        - Order 1 derivative, :math:`\psi_{\mathbf{R}_j}(k, \vec{z}_i) = \sum_{i\in\mathbf{R}(v_j)} z_{ik} \cdot g(\vec{z}_i)`
        - Order 2 derivative, :math:`\psi_{\mathbf{R}_j}(h,~k,~\vec{z}_i) = \sum_{i\in\mathbf{R}(v_j)} z_{ik} \cdot z_{ih} \cdot g(\vec{z}_i)`
    See :code:`Cox(*args)._psi_efron(*args)`
"""


@final
class CoxPartialLifetimeLikelihood(
    MaximumLikehoodOptimizer[LinearCovarEffect, CoxData]
):
    data: CoxData
    # https://github.com/microsoft/pyright/issues/6564
    model: LinearCovarEffect
    scipy_method = "trust-exact"

    def __init__(
        self,
        covar_effect: LinearCovarEffect,
        data: CoxData,
    ):
        self.model = copy.deepcopy(covar_effect)
        self.data = data

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def _initialize_model(self) -> LinearCovarEffect:
        self.model.params = np.random.random(self.data.covar.shape[1])
        return self.model

    @override
    def _get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.model.nb_params, -np.inf),
            np.full(self.model.nb_params, np.inf),
        )

    @override
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        self.model.params = params
        return -(
            np.log(self.model.g(self.data.ordered_event_covar)).sum()
            - np.log(psi(self.model, self.data)).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.params = params  # changes model params

        return -(
            self.data.ordered_event_covar.sum(axis=0)
            - (psi(self.model, self.data, order=1) / psi(self.model, self.data)).sum(
                axis=0
            )
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> ToFloat2D:
        self.model.params = params  # changes model params

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
    MaximumLikehoodOptimizer[LinearCovarEffect, CoxData]
):
    model: LinearCovarEffect
    data: CoxData
    s_j: NDArray[np.float64]
    scipy_method = "trust-exact"

    def __init__(
        self,
        covar_effect: LinearCovarEffect,
        data: CoxData,
    ):
        self.model = copy.deepcopy(covar_effect)
        self.data = data
        self.s_j = np.dot(self.data.death_set, self.data.covar)

    @property
    @override
    def nb_observations(self) -> int:
        return len(self.data.time)

    @override
    def _initialize_model(self) -> LinearCovarEffect:
        self.model.params = np.random.random(self.data.covar.shape[1])
        return self.model

    @override
    def _get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.model.nb_params, -np.inf),
            np.full(self.model.nb_params, np.inf),
        )

    @override
    def negative_log(self, params: Array1D[np.float64]) -> ToFloat:
        self.model.params = params  # changes model params

        return -(
            np.log(self.model.g(self.s_j)).sum()
            - (
                self.data.event_count[:, None] * np.log(psi(self.model, self.data))
            ).sum()
        )

    def jac_negative_log(self, params: Array1D[np.float64]) -> Array1D[np.float64]:
        self.model.params = params  # changes model params

        return -(
            self.s_j.sum(axis=0)
            - (
                self.data.event_count[:, None]
                * (psi(self.model, self.data, order=1) / psi(self.model, self.data))
            ).sum(axis=0)
        )

    def hess_negative_log(self, params: Array1D[np.float64]) -> ToFloat2D:
        self.model.params = params  # changes model params

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
    MaximumLikehoodOptimizer[LinearCovarEffect, CoxData]
):
    model: LinearCovarEffect
    data: CoxData
    s_j: NDArray[np.float64]
    discount_rates: NDArray[np.float64]
    discount_rates_mask: NDArray[np.bool_]
    scipy_method = "trust-exact"

    def __init__(
        self,
        covar_effect: LinearCovarEffect,
        data: CoxData,
    ):
        self.model = copy.deepcopy(covar_effect)
        self.data = data
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

    @override
    def _initialize_model(self) -> LinearCovarEffect:
        self.model.params = np.random.random(self.data.covar.shape[1])
        return self.model

    @override
    def _get_params_bounds(self) -> Bounds:
        return Bounds(
            np.full(self.model.nb_params, -np.inf),
            np.full(self.model.nb_params, np.inf),
        )

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
        """

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
    def negative_log(
        self,
        params: NDArray[np.float64],
    ) -> float:
        self.model.params = params  # changes model params

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
        self.model.params = params  # changes model params
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

    def hess_negative_log(self, params: Array1D[np.float64]) -> ToFloat2D:
        self.model.params = params  # changes model params

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
