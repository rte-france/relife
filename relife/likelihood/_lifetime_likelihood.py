from __future__ import annotations

from typing import TYPE_CHECKING, Any, final

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.utils import reshape_1d_arg

from ._base import Likelihood

if TYPE_CHECKING:
    from relife.lifetime_model._base import FittableParametricLifetimeModel
    from relife.lifetime_model._semi_parametric import Cox

__all__ = ["DefaultLifetimeLikelihood", "IntervalLifetimeLikelihood", "PartialLifetimeLikelihood"]


@final
class DefaultLifetimeLikelihood(Likelihood):

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
        super().__init__(model, time=time, event=event, entry=entry) # TODO: il faudrait pouvoir extraire covar de model_args....
        self.params = self.model.get_initial_params(time, model_args)

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
        self._nb_observations = len(time)

    @property
    @override
    def nb_observations(self):
        return self._nb_observations

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


@final
class IntervalLifetimeLikelihood(Likelihood):
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
    ):
        super().__init__(model)
        self.params = self.model._get_initial_params(time_sup, model_args)
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

    @property
    @override
    def nb_observations(self):
        return self._nb_observations

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


class PartialLifetimeLikelihood(Likelihood):

    def __init__(
            self,
            model: Cox,
            time: NDArray[np.float64],
            covar: NDArray[np.float64],
            event: NDArray[np.bool_] | None = None,
            entry: NDArray[np.float64] | None = None,
    ):
        super().__init__(model, time=time, covar=covar, event=event, entry=entry)
        self.params = self.model.get_initial_params(time, covar)

        time = reshape_1d_arg(time)
        event = reshape_1d_arg(event) if event is not None else np.ones_like(time, dtype=np.bool_)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time, dtype=np.float64)
        sizes = [len(x) for x in (time, event, entry, covar) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {tuple(sizes)}"
            )

        (
            ordered_event_time,    # uncensored sorted untied times
            ordered_event_index,
            self._event_count,
        ) = np.unique(
            time[event == 1],
            return_index=True,
            return_counts=True,
        )
        # here risk_set is mask array on time
        # left truncated & right censored
        self._risk_set = np.logical_and(
            (
                    np.vstack([entry[:, 0]] * len(ordered_event_time))
                    < np.hstack([ordered_event_time[:, None]] * len(time))
            ),
            (
                    np.hstack([ordered_event_time[:, None]] * len(time))
                    <= np.vstack([time[:, 0]] * len(ordered_event_time))
            ),
        )

        self._death_set = np.vstack([time[:, 0] * event[:, 0]] * len(ordered_event_time)) == np.hstack(
            [ordered_event_time[:, None]] * len(time)
        )

        self._covar = covar
        self._ordered_event_covar = covar[event[:, 0] == 1][ordered_event_index]

        self._nb_observations = len(time)

        if (self._event_count > 3).any():
            self.set_method("efron")
        elif (self._event_count <= 3).all() and (2 in self._event_count):
            self.set_method("breslow")
        else:
            self.set_method("cox")

    @property
    def nb_observations(self):
        return self._nb_observations

    def set_method(self, method: str) -> None:
        r"""Manually specify method used to compute partial likelihood and its derivates. By default :code:`method` is set automatically.

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
        if method.lower() == "efron":
            self.method = method
            self._compute_s_j()
            self._compute_efron_discount_rates()
        elif method.lower() == "breslow":
            self.method = method
            self._compute_s_j()
            self._discount_rates = None
            self._discount_rates_mask = None
        elif method.lower() == "cox":
            self.method = method
            self._s_j = None
            self._discount_rates = None
            self._discount_rates_mask = None
        else:
            raise ValueError(f"method allowed are efron, breslow or cox. Not {method}")

    def _compute_s_j(self) -> None:
        """s_j : [m, p]"""
        self._s_j = np.dot(self._death_set, self._covar)

    def _compute_efron_discount_rates(self) -> None:
        """
        discount_rates : [m, max(event_count)] or [m, max(event_count)]
        discount_rates_mask : [m, max(event_count)] or [m, max(event_count)]
        """

        self._discount_rates = (
                np.vstack([np.arange(self._event_count.max())] * len(self._event_count))
                / self._event_count[:, None]
        )
        self._discount_rates_mask = np.where(self._discount_rates < 1, 1, 0)

    def _psi(self, on: str = "risk", order: int = 0) -> np.ndarray:
        r"""Psi formula used for likelihood computations

        Args:
            on (str, optional): "risk" or "death". Defaults to "risk". If "death", sum is applied on death set.
            order (int, optional): order derivatives with respect to params. Defaults to 0.

        Returns:
            np.ndarray: psi formulation
            If order 0, shape [m, 1]
            If order 1, shape [m, p]
            If order 2, shape [m, p, p]
        """
        if on == "risk":
            i_set = self._risk_set
        elif on == "death":
            i_set = self._death_set
        else:
            raise ValueError(f"'on' allowed values are 'risk' and 'death', not {on}")

        if order == 0:
            # shape [m]
            return np.dot(i_set, self.model.covar_effect.g(self._covar))
        elif order == 1:
            # shape [m, p]
            return np.dot(i_set, self._covar * self.model.covar_effect.g(self._covar))
        elif order == 2:
            # shape [m, p, p]
            return np.tensordot(
                i_set[:, :None],
                self._covar[:, None]
                * self._covar[:, :, None]
                * self.model.covar_effect.g(self._covar)[:, :, None],
                axes=1,
            )

    def _psi_efron(self, order: int = 0) -> np.ndarray:
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
                    self._psi() * self._discount_rates_mask
                    - self._psi(on="death")
                    * self._discount_rates
                    * self._discount_rates_mask
            )
        elif order == 1:
            # shape [m, max(d_j), p]
            return (
                    self._psi(order=1)[:, None, :]
                    * self._discount_rates_mask[:, :, None]
                    - self._psi(on="death", order=1)[:, None, :]
                    * (self._discount_rates * self._discount_rates_mask)[:, :, None]
            )
        elif order == 2:
            # shape [m, max(d_j), p, p]
            return (
                    self._psi(order=2)[:, None, :]
                    * self._discount_rates_mask[:, :, None, None]
                    - self._psi(on="death", order=2)[:, None, :]
                    * (self._discount_rates * self._discount_rates_mask)[:, :, None, None]
            )

    def negative_log(
            self,
            params: NDArray[np.float64],
        ) -> float:
        """Compute negative log partial likelihood depending on method used (cox, breslow or efron)

        Returns:
            float : negative log partial likelihood at params
        """
        self.params = params  # changes model params

        if self.method == "cox":
            neg_L = -(
                    np.log(self.model.covar_effect.g(self._ordered_event_covar)).sum()
                    - np.log(self._psi()).sum()
            )
        elif self.method == "breslow":
            neg_L = -(
                    np.log(self.model.covar_effect.g(self._s_j)).sum()
                    - (self._event_count[:, None] * np.log(self._psi())).sum()
            )
        elif self.method == "efron":
            # .sum(axis=1, keepdims=True) --> sum on alpha to d_j
            # .sum() --> sum on j
            # using where in np.log allows to avoid 0. masked elements
            m = self._psi_efron()
            neg_L = -(
                    np.log(self.model.covar_effect.g(self._s_j)).sum()
                    - np.log(m, out=np.zeros_like(m), where=(m != 0))
                    .sum(axis=1, keepdims=True)
                    .sum()
            )
        return neg_L

    def jac_negative_log(
            self,
            params: NDArray[np.float64],
        ) -> np.ndarray:
        """Compute Jacobian of the negative log partial likelihood depending on method used (cox, breslow or efron)

        Returns:
            np.ndarray: jacobian vector
        """
        self.params = params  # changes model params

        if self.method == "cox":
            return -(
                    self._ordered_event_covar.sum(axis=0)
                    - (self._psi(order=1) / self._psi()).sum(axis=0)
            )
        elif self.method == "breslow":
            return -(
                    self._s_j.sum(axis=0)
                    - (
                            self._event_count[:, None]
                            * (self._psi(order=1) / self._psi())
                    ).sum(axis=0)
            )
        elif self.method == "efron":
            # .sum(axis=1) --> sum on alpha to d_j
            # .sum(axis=0) --> sum on j
            # using where in np.divide allows to avoid 0. masked elements
            a = self._psi_efron(order=1)
            b = self._psi_efron()[:, :, None]
            return -(
                    self._s_j.sum(axis=0)
                    - np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
                    .sum(axis=1)
                    .sum(axis=0)
            )

    def hess_negative_log(
            self,
            params: NDArray[np.float64],
        ) -> np.ndarray:
        """Compute Hessian of the negative log partial likelihood depending on method used (cox, breslow or efron)

        Returns:
            np.ndarray: hessian matrix
        """
        self.params = params  # changes model params

        if self.method == "cox" or self.method == "breslow":
            psi_order_0 = self._psi()
            psi_order_1 = self._psi(order=1)

            hessian_part_1 = self._psi(order=2) / psi_order_0[:, :, None]
            # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

            hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
                    psi_order_1 / psi_order_0
            )[:, :, None]
            # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

            if self.method == "cox":
                return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)
            elif self.method == "breslow":
                return (self._event_count[:, None, None] * hessian_part_1).sum(
                    axis=0
                ) - (self._event_count[:, None, None] * hessian_part_2).sum(axis=0)

        elif self.method == "efron":
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
                    np.divide(
                        psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0)
                    )[:, :, None, :]
                    * (
                        np.divide(
                            psi_order_1, b, out=np.zeros_like(psi_order_1), where=(b != 0)
                        )
                    )[:, :, :, None]
            )
            hessian_part_2 = hessian_part_2.sum(axis=1)

            return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)