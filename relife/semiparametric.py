import numpy as np
import warnings
from dataclasses import dataclass
from typing import Tuple, Union

from scipy.optimize import minimize
from scipy.stats import chi2, norm
from scipy import linalg
from .nonparametric import NelsonAalen
import matplotlib.pyplot as plt


def _nearest_1dinterp(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Returns x nearest interpolation based on xp and yp data points
    xp has to be monotonically increasing

    Args:
        x (np.ndarray): 1d x coordinates to interpolate
        xp (np.ndarray): 1d known x coordinates
        yp (np.ndarray): 1d known y coordinates

    Returns:
        np.ndarray: interpolation values of x
    """
    spacing = np.diff(xp) / 2
    xp = xp + np.hstack([spacing, spacing[-1]])
    yp = np.concatenate([yp, yp[-1, None]])
    return yp[np.searchsorted(xp, x)]


class Cox:
    r"""Cox regression model

    :math:`h(t,~\vec{z}) = h_0(t)\times e^{\vec{\beta}^\intercal\cdot\vec{z}} = h_0(t) \times g(\vec{z})`

    **Only right censored and left truncated times are allowed**. In case of tied events, two partial likelihood approximations are implemented : Breslow and Efron.

    Args:
        time (np.ndarray): shape :math:`n` - Age of the assets, :math:`u_i`.
        covar (np.ndarray): shape :math:`(n, p)` - Covariates, :math:`\vec{z}_i`.
        event (np.ndarray): shape :math:`n` - Type of event, :math:`\delta_i`, by default None.
        entry (np.ndarray): shape :math:`n` - Age of assets at the beginning of the observation period (left truncation), by default None.

    Attributes:
        ordered_event_time (np.ndarray): shape :math:`m` - Ordered distinct ages of the assets, :math:`v_j`.
        event_count (np.ndarray): shape :math:`m` - Number of death at each ordered distinct ages, :math:`d_j`.
        ordered_event_covar (np.ndarray) shape :math:`(m, p)` - Ordered distinct covariates :math:`\vec{z}_j` at :math:`v_j` when they are no ties (:math:`\vec{s}_j` is used for tied events)
        risk_set (np.ndarray) shape :math:`(m, n)` - Set of all assets :math:`i` at risk just prior to :math:`v_j`. It corresponds to :math:`\mathbf{R}_j`
        death_set (np.ndarray) shape :math:`(m, n)` - Set of all assets :math:`i` who die at :math:`v_j`. Only used for Breslow and Efron method when events are tied. It corresponds to :math:`\mathbf{D}_j`
        tied_event (bool) True if event times occur simultaneously.

    Examples::
        >>> cox = Cox()
    """

    def __init__(
        self,
        time: np.ndarray,
        covar: Union[np.ndarray, list, tuple],
        event: np.ndarray = None,
        entry: np.ndarray = None,
    ):
        assert isinstance(time, np.ndarray), "time must be np.ndarray"
        assert (
            isinstance(event, np.ndarray) or event is None
        ), "event must be np.ndarray or None"
        assert (
            isinstance(entry, np.ndarray) or entry is None
        ), "entry must be np.ndarray or None"
        assert isinstance(covar, np.ndarray), "covar must be np.ndarray, list or tuple"

        self.time = time
        # self.covar = covar
        self.event = event
        self.entry = entry
        self.param = None

        if self.event is None:
            self.event = np.ones_like(self.time, int)
        if self.entry is None:
            self.entry = np.zeros_like(self.time, float)

        assert (
            len(covar.shape) <= 2
        ), f"covar has shape {covar.shape} but must be 1d or 2d"
        if len(covar.shape) == 1:
            covar = covar[:, None]
        assert covar.shape[0] == len(
            self.time
        ), f"covar must have the same length than time"
        self.covar = covar

        assert (
            len(self.time.shape) == 1
            and len(self.event.shape) == 1
            and len(self.covar.shape) == 2
        ), "time and event must be 1d array and covar 2d array"
        assert (
            len(self.time) == len(self.covar) == len(self.event)
        ), "conflicting input data dimensions"

        (
            self.ordered_event_time,
            sorted_uncensored_i,
            self.event_count,
        ) = np.unique(
            self.time[self.event == 1],  # uncensored sorted times
            return_index=True,
            return_counts=True,
        )

        self.ordered_event_covar = self.covar[self.event == 1][sorted_uncensored_i]

        # here risk_set is mask array on time
        # left truncated & right censored
        self.risk_set = np.logical_and(
            (
                np.vstack([self.entry] * len(self.ordered_event_time))
                < np.hstack([self.ordered_event_time[:, None]] * len(self.time))
            ),
            (
                np.hstack([self.ordered_event_time[:, None]] * len(self.time))
                <= np.vstack([self.time] * len(self.ordered_event_time))
            ),
        )

        self.death_set = np.vstack(
            [self.time * self.event] * len(self.ordered_event_time)
        ) == np.hstack([self.ordered_event_time[:, None]] * len(self.time))

        if (self.event_count > 3).any():
            self.set_method("efron")
        elif (self.event_count <= 3).all() and (2 in self.event_count):
            self.set_method("breslow")
        else:
            self.set_method("cox")
        self.tied_event = (self.event_count > 1).any()

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
            self.discount_rates = None
            self.discount_rates_mask = None
        elif method.lower() == "cox":
            self.method = method
            self.s_j = None
            self.discount_rates = None
            self.discount_rates_mask = None
        else:
            raise ValueError(f"method allowed are efron, breslow or cox. Not {method}")

    @staticmethod
    def _g(covar: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """:math:`e^{\vec{\beta}^\intercal \cdot \vec{z}}`"""

        return np.exp(np.dot(covar, beta[:, None]))

    @staticmethod
    def _log_g(covar: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """math:`\vec{\beta}^\intercal \cdot \vec{z}`"""

        return np.dot(covar, beta[:, None])

    def _compute_s_j(self) -> None:
        """s_j : [m, p]"""
        self.s_j = np.dot(self.death_set, self.covar)

    def _compute_efron_discount_rates(self) -> None:
        """
        discount_rates : [m, max(event_count)] or [m, max(event_count)]
        discount_rates_mask : [m, max(event_count)] or [m, max(event_count)]
        """

        self.discount_rates = (
            np.vstack([np.arange(self.event_count.max())] * len(self.event_count))
            / self.event_count[:, None]
        )
        self.discount_rates_mask = np.where(self.discount_rates < 1, 1, 0)

    def _psi(self, beta: np.ndarray, on: str = "risk", order: int = 0) -> np.ndarray:
        r"""Psi formula used for likelihood computations

        Args:
            beta (np.ndarray): parameter vector
            on (str, optional): "risk" or "death". Defaults to "risk". If "death", sum is applied on death set.
            order (int, optional): order derivatives with respect to beta. Defaults to 0.

        Returns:
            np.ndarray: psi formulation
            If order 0, shape [m, 1]
            If order 1, shape [m, p]
            If order 2, shape [m, p, p]
        """
        if on == "risk":
            i_set = self.risk_set
        elif on == "death":
            i_set = self.death_set

        if order == 0:
            # shape [m]
            return np.dot(i_set, Cox._g(self.covar, beta))
        elif order == 1:
            # shape [m, p]
            return np.dot(i_set, self.covar * Cox._g(self.covar, beta))
        elif order == 2:
            # shape [m, p, p]
            return np.tensordot(
                i_set[:, :None],
                self.covar[:, None]
                * self.covar[:, :, None]
                * Cox._g(self.covar, beta)[:, :, None],
                axes=1,
            )

    def _psi_efron(self, beta: np.ndarray, order: int = 0) -> np.ndarray:
        """Psi formula for Efron method

        Args:
            beta (np.ndarray): parameter vector
            order (int, optional): order derivatives with respect to beta. Defaults to 0.

        Returns:
            np.ndarray: psi formulation for Efron method
            If order 0, shape [m, max(d_j)]
            If order 1, shape [m, max(d_j), p]
            If order 2, shape [m, max(d_j), p, p]
        """

        if order == 0:
            # shape [m, max(d_j)]
            return (
                self._psi(beta) * self.discount_rates_mask
                - self._psi(beta, on="death")
                * self.discount_rates
                * self.discount_rates_mask
            )
        elif order == 1:
            # shape [m, max(d_j), p]
            return (
                self._psi(beta, order=1)[:, None, :]
                * self.discount_rates_mask[:, :, None]
                - self._psi(beta, on="death", order=1)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None]
            )
        elif order == 2:
            # shape [m, max(d_j), p, p]
            return (
                self._psi(beta, order=2)[:, None, :]
                * self.discount_rates_mask[:, :, None, None]
                - self._psi(beta, on="death", order=2)[:, None, :]
                * (self.discount_rates * self.discount_rates_mask)[:, :, None, None]
            )

    def _negative_log_partial_likelihood(self, beta: np.ndarray) -> float:
        """Compute negative log partial likelihood depending on method used (cox, breslow or efron)

        Args:
            beta (np.ndarray): parameter vector

        Returns:
            float : negative log partial likelihood at beta
        """
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.covar.shape[1], "conflicting beta dimension with covar"

        # neg_L_cox == neg_L_breslow == neg_L_efron if (not self.tied_events)
        if self.method == "cox":
            return -(
                (Cox._log_g(self.ordered_event_covar, beta)).sum()
                - np.log(self._psi(beta)).sum()
            )
        elif self.method == "breslow":
            return -(
                (Cox._log_g(self.s_j, beta)).sum()
                - (self.event_count[:, None] * np.log(self._psi(beta))).sum()
            )
        elif self.method == "efron":
            # .sum(axis=1, keepdims=True) --> sum on alpha to d_j
            # .sum() --> sum on j
            # using where in np.log allows to avoid 0. masked elements
            m = self._psi_efron(beta)
            neg_L_efron = -(
                (Cox._log_g(self.s_j, beta)).sum()
                - np.log(m, out=np.zeros_like(m), where=(m != 0))
                .sum(axis=1, keepdims=True)
                .sum()
            )
            return neg_L_efron

    def _jac(self, beta: np.ndarray) -> np.ndarray:
        """Compute Jacobian of the negative log partial likelihood depending on method used (cox, breslow or efron)

        Args:
            beta (np.ndarray): parameter vector

        Returns:
            np.ndarray: jacobian vector
        """
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.covar.shape[1], "conflicting beta dimension with covar"

        if self.method == "cox":
            return -(
                self.ordered_event_covar.sum(axis=0)
                - (self._psi(beta, order=1) / self._psi(beta)).sum(axis=0)
            )
        elif self.method == "breslow":
            return -(
                self.s_j.sum(axis=0)
                - (
                    self.event_count[:, None]
                    * (self._psi(beta, order=1) / self._psi(beta))
                ).sum(axis=0)
            )
        elif self.method == "efron":
            # .sum(axis=1) --> sum on alpha to d_j
            # .sum(axis=0) --> sum on j
            # using where in np.divide allows to avoid 0. masked elements
            a = self._psi_efron(beta, order=1)
            b = self._psi_efron(beta)[:, :, None]
            return -(
                self.s_j.sum(axis=0)
                - np.divide(a, b, out=np.zeros_like(a), where=(b != 0))
                .sum(axis=1)
                .sum(axis=0)
            )

    def _hess(self, beta: np.ndarray) -> np.ndarray:
        """Compute Hessian of the negative log partial likelihood depending on method used (cox, breslow or efron)

        Args:
            beta (np.ndarray): parameter vector

        Returns:
            np.ndarray: hessian matrix
        """
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.covar.shape[1], "conflicting beta dimension with covar"

        if self.method == "cox" or self.method == "breslow":
            psi_order_0 = self._psi(beta)
            psi_order_1 = self._psi(beta, order=1)

            hessian_part_1 = self._psi(beta, order=2) / psi_order_0[:, :, None]
            # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

            hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
                psi_order_1 / psi_order_0
            )[:, :, None]
            # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

            if self.method == "cox":
                return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)
            elif self.method == "breslow":
                return (self.event_count[:, None, None] * hessian_part_1).sum(
                    axis=0
                ) - (self.event_count[:, None, None] * hessian_part_2).sum(axis=0)

        elif self.method == "efron":
            psi_order_0 = self._psi_efron(beta)
            psi_order_1 = self._psi_efron(beta, order=1)

            # .sum(axis=1) --> sum on alpha to d_j
            # using where in np.divide allows to avoid 0. masked elements
            a = self._psi_efron(beta, order=2)
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

    def fit(self, method: str = "trust-exact") -> np.ndarray:
        """
        Fit the covariate effect to time, covar, event and entry arrays.

        References:
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html#scipy.optimize.minimize
        """
        opt = minimize(
            fun=self._negative_log_partial_likelihood,
            x0=np.random.random(self.covar.shape[1]),
            method=method,
            jac=self._jac,
            hess=self._hess,
        )
        self.param = opt.x

    @property
    def AIC(self) -> float:
        """Return AIC value divided by 2

        Args:
            beta (np.ndarray): estimates of parameter vector, shape p

        Returns:
            float: AIC value divided by 2
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling AIC")
            return None
        return self._negative_log_partial_likelihood(self.param) + len(self.param)

    @property
    def information(self) -> np.ndarray:
        """Return Fisher information matrix

        Returns:
            np.ndarray: Fisher information matrix
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling information")
            return None
        return self._hess(self.param)

    @property
    def var(self) -> np.ndarray:
        """Return estimated covariance matrix of estimated parameters

        Returns:
            np.ndarray: covariance matrix, shape (p,p)
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling var")
            return None
        return linalg.inv(self.information)

    @property
    def std(self) -> np.ndarray:
        """Return estimated standard error of estimated parameters

        Returns:
            np.ndarray: standard error, shape p
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling std")
            return None
        return np.sqrt(np.diag(self.var))

    def chf0(self, conf_int: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Knowing estimates of beta, computes the cumulative baseline hazard rate estimator and its confidence interval (optional)

        Args:
            beta (np.ndarray): estimates of parameter vector, shape p
            conf_int (bool, optional): If true returns estimated confidence interval. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: values of chf0 estimator and its confidence interval
            at 95% level. Arrays are of size :math:`m`

        Examples:
            >>> beta = my_cox.fit()
            >>> chf0_values, conf_int = my_cox.chf0(beta, conf_int=True)
            >>> plt.step(my_cox.ordered_event_time, chf0_values, where="post")
            >>> plt.fill_between(my_cox.ordered_event_time, conf_int[:, 0], conf_int[:, 1], alpha=0.25, step="post")
            >>> plt.show()
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling chf0")
            return None
        values = np.cumsum(self.event_count[:, None] / self._psi(self.param))
        if conf_int:
            var = np.cumsum(self.event_count[:, None] / self._psi(self.param) ** 2)
            conf_int = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )
            return values, conf_int
        else:
            return values

    def sf0(self, conf_int: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Knowing estimates of beta, computes the baseline survival function and its confidence interval (optional)

        Args:
            beta (np.ndarray): estimates of parameter vector, shape p
            conf_int (bool, optional): If true returns estimated confidence interval. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]: values of chf0 estimator and its confidence interval
            at 95% level. Arrays are of size :math:`m`

        Examples:
            >>> beta = my_cox.fit()
            >>> sf0_values, conf_int = my_cox.sf0(beta, conf_int=True)
            >>> plt.step(my_cox.ordered_event_time, sf0_values, where="post")
            >>> plt.fill_between(my_cox.ordered_event_time, conf_int[:, 0], conf_int[:, 1], alpha=0.25, step="post")
            >>> plt.show()
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling sf0")
            return None
        elif conf_int:
            chf0, chf0_conf_int = -self.chf(conf_int=True)
            return np.exp(-chf0), np.exp(-chf0_conf_int)
        else:
            return np.exp(-self.chf())

    def sf(
        self, covar: np.ndarray, conf_int: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Knowing estimates of beta, computes the sf estimator and confidence interval (optional)

        Args:
            covar (np.ndarray): one vector of covariate values, shape p
            conf_int (bool, optional): If true returns estimated confidence interval. Defaults to False.

        Returns:
            Tuple[np.ndarray, np.ndarray]:  values of sf estimator and its confidence interval
            at 95% level. Arrays are of size :math:`m`

        Examples:
            >>> beta = my_cox.fit()
            >>> sf_values, conf_int = my_cox.sf(beta, my_cox.covar[21]) # sf of the 22th asset
            >>> plt.step(my_cox.ordered_event_time, sf_values, where="post")
            >>> plt.fill_between(my_cox.ordered_event_time, conf_int[:, 0], conf_int[:, 1], alpha=0.25, step="post")
            >>> plt.show()
        """
        if self.param is None:
            warnings.warn("cox model has to be fitted before calling sf")
            return None
        values = np.exp(-self.chf()) ** Cox._g(covar, self.param)
        if conf_int:
            psi = self._psi(self.param)
            psi_order_1 = self._psi(self.param, order=1)
            d_j_on_psi = self.event_count[:, None] / psi
            information_matrix = self._hess(self.param)
            inverse_information_matrix = linalg.inv(information_matrix)

            q3 = np.cumsum((psi_order_1 / psi - covar) * d_j_on_psi, axis=0)  # [m, p]
            q2 = np.squeeze(
                np.matmul(
                    q3[:, None, :],
                    np.matmul(inverse_information_matrix[None, :, :], q3[:, :, None]),
                )
            )  # m
            q1 = np.cumsum(d_j_on_psi * (1 / psi))

            var = (values**2) * (q1 + q2)

            conf_int = np.hstack(
                [
                    values[:, None]
                    + np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                    values[:, None]
                    - np.sqrt(var)[:, None] * norm.ppf(0.05 / 2, loc=0, scale=1),
                ]
            )

            return values, conf_int
        else:
            return values

    # def rr(self, beta: np.ndarray) -> np.ndarray:
    #     """Return the relative risks

    #     Args:
    #         beta (np.ndarray): estimates of parameter vector, shape p

    #     Returns:
    #         np.ndarray: relative risks, shape (p,p)
    #     """
    #     if self.param is None:
    #         warnings.warn("cox model has to be fitted before calling sf0")
    #         return None
    #     return np.dot(np.exp(beta)[:, None], (1 / np.exp(beta))[None, :])


def cox_snell_residuals_plot(cox: Cox) -> None:
    """Graphical check of the overall fit of the cox model

    Args:
        cox (Cox): Cox instance model

    Examples:
        >>> cox_snell_residuals_plot(cox_model)
    """
    
    # compute cox_snell residuals
    chf0_values = _nearest_1dinterp(cox.time, cox.ordered_event_time, cox.chf0())
    residuals = chf0_values * np.squeeze(Cox._g(cox.covar, cox.param))

    # compute chf values of residuals
    nelson_aalen_estimator = NelsonAalen()
    # nelson_aalen_estimator.fit(residuals, cox.event, cox.entry)
    nelson_aalen_estimator.fit(residuals, cox.event)

    chf_of_residuals = _nearest_1dinterp(
        residuals, nelson_aalen_estimator.timeline, nelson_aalen_estimator.chf
    )

    ordered_residuals_index = np.argsort(residuals)

    # plot results
    fig, ax = plt.subplots()
    ax.step(
        residuals[ordered_residuals_index],
        chf_of_residuals[ordered_residuals_index],
        where="post",
    )
    ax.plot(
        [0, np.max(residuals)],
        [0, np.max(residuals)],
        c="black",
        linestyle="--",
    )
    ax.set_xlabel("Residual")
    ax.set_ylabel("Estimated Cumulative Hazard Rates")
    fig.tight_layout()
    plt.show()


def cox_proportionality_effect_plot(cox: Cox, nb_strata: int = 4, andersen: bool = False) -> None:
    """Graphical checks of the proportional effects of covariates assumption

    Args:
        cox (Cox): Cox instance model 
        nb_strata (int, optional): number of strata used for covariate values. Defaults to 4.
        andersen (bool, optional): If True, Andersen plots are used. Defaults to False, then difference of log cumulative hazard rates is used

    Raises:
        ValueError: the number of strata must not be too high to keep enough data per stratum

    Examples:
        >>> cox_proportionality_effect_plot(cox_model, nb_strata=4)
        >>> cox_proportionality_effect_plot(cox_model, nb_strata=4, andersen=True)
    """

    # set figure grid
    if cox.covar.shape[1] % 2 == 0:
        fig, ax = plt.subplots(
            cox.covar.shape[1] // 2, 2, layout="constrained", squeeze=False
        )
    else:
        fig, ax = plt.subplots(
            cox.covar.shape[1] // 2 + 1, 2, layout="constrained", squeeze=False
        )
        fig.delaxes(ax[cox.covar.shape[1] // 2, 1])

    timeline = np.sort(cox.time)

    # iterate through each covariate
    for covar_index in range(cox.covar.shape[1]):
        # stratify continuous covar into categorical values
        # bins are the q-th quantile
        # if covar values are already categorical, they remain the same
        bins = np.quantile(
            cox.covar[:, covar_index], q=np.cumsum(np.ones(nb_strata) / nb_strata)
        )
        categorical_values = (
            np.digitize(cox.covar[:, covar_index], bins, right=True) + 1
        )

        covar_strata = np.copy(cox.covar)
        covar_strata[:, covar_index] = categorical_values
        chf0_strata = np.empty((len(np.unique(categorical_values)), len(timeline)))

        # compute chf0 for each stratum
        for i, value in enumerate(np.unique(categorical_values)):
            value_index = np.where(categorical_values == value)[0]
            if len(value_index) == 1:
                raise ValueError(
                    f"Nb of strata is too high and {i}-th stratum only corresponds to one value. Decrease nb_strata value"
                )

            cox_at_value = Cox(
                np.copy(cox.time[value_index]),
                covar_strata[value_index, :],
                np.copy(cox.event[value_index]),
                np.copy(cox.entry[value_index]),
            )
            cox_at_value.fit()

            chf0_strata[i] = _nearest_1dinterp(
                timeline, cox_at_value.ordered_event_time, cox_at_value.chf0()
            )

        if andersen:
            # standardize chf values to compare strata
            chf0_strata = (
                chf0_strata - chf0_strata.mean(axis=1)[:, None]
            ) / chf0_strata.std(axis=1)[:, None]
            # should be y = x line
            ax[covar_index // 2, covar_index % 2].plot(
                [0, np.max(chf0_strata[0])],
                [0, np.max(chf0_strata[0])],
                c="black",
                linestyle="--",
            )
            for i in range(1, chf0_strata.shape[0]):
                ax[covar_index // 2, covar_index % 2].set_title(
                    f"Covar Z_{covar_index+1}"
                )
                ax[covar_index // 2, covar_index % 2].step(
                    chf0_strata[0],
                    chf0_strata[i],
                    where="post",
                    label=f"strata {i + 1} vs. strata 1",
                )
                # ax[covar_index // 2, covar_index % 2].set_title(f"Covar Z_{covar_index+1}")
                ax[covar_index // 2, covar_index % 2].legend()
                fig.suptitle(
                    "Andersen plots of standardized cumulative hazard rates strata"
                )

        else:
            log_chf0_diff = np.log(
                chf0_strata[1:] / np.full_like(chf0_strata[1:], chf0_strata[0])
            )
            # standardize chf values to compare strata
            log_chf0_diff = (
                log_chf0_diff - log_chf0_diff.mean(axis=1)[:, None]
            ) / log_chf0_diff.std(axis=1)[:, None]
            # should be horizontal line y = 0 if centered
            ax[covar_index // 2, covar_index % 2].plot(
                [0, timeline[-1]], [0, 0], c="black", linestyle="--"
            )
            for i in range(log_chf0_diff.shape[0]):
                ax[covar_index // 2, covar_index % 2].step(
                    timeline,
                    log_chf0_diff[i],
                    where="post",
                    label=f"log (strata {i + 2} / strata 1)",
                )
                ax[covar_index // 2, covar_index % 2].set_title(
                    f"Covar Z_{covar_index+1}"
                )
                ax[covar_index // 2, covar_index % 2].set_xlabel("Time on study")
                ax[covar_index // 2, covar_index % 2].legend()
                fig.suptitle(
                    "Difference in standardized log cumulative hazard rates strata"
                )

    plt.show()


def cox_wald_test(cox: Cox, c: np.ndarray = None) -> Tuple[float, float]:
    """Perform Wald's test (testing nullity of covariate effect)

    Args:
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating which covar coordinate is 0 in the null hypothesis
        Defaults to None, then the null hypothesis corresponds to null effect of all covariates

    Returns:
        Tuple[float, float]: test value and its corresponding pvalue

    Examples:
        >>> my_cox.fit() # beta is of shape (4,)
        >>> chi-square, pvalue = my_cox.wald_test(my_cox, [0, 1, 1, 1]) # only the first covariate is tested as 0
        >>> chi-square, pvalue = my_cox.wald_test(my_cox, [0, 0, 0, 1]) # the first three covariates are tested as 0
    """

    if isinstance(c, list):
        assert cox.covar.shape[-1] == len(c)
        c = np.array(c)
    elif isinstance(c, np.ndarray):
        assert len(c.shape) == 1
        assert cox.covar.shape[-1] == c.shape[-1]

    if c is None:
        # null hypothesis is beta = 0
        ch2 = np.dot(cox.param, np.dot(cox.information, cox.param))
        pval = chi2.sf(ch2, df=cox.covar.shape[-1])
        return round(ch2, 6), round(pval, 6)
    else:
        # local test
        other_covar = np.where(c == 0)[0]

        ch2 = np.dot(
            cox.param[other_covar],
            np.dot(
                linalg.inv(
                    linalg.inv(cox.information)[np.ix_(other_covar, other_covar)]
                ),
                cox.param[other_covar],
            ),
        )
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)


def cox_likelihood_ratio_test(cox: Cox, c: np.ndarray = None) -> Tuple[float, float]:
    """Perform likelihood ratio test (testing nullity of covariate effect)

    Args:
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating which covar coordinate is 0 in the null hypothesis
        Defaults to None, then the null hypothesis corresponds to null effect of all covariates

    Returns:
        Tuple[float, float]: test value and its corresponding pvalue

    Examples:
        >>> my_cox.fit() # beta is of shape (4,)
        >>> chi-square, pvalue = my_cox.likelihood_ratio_test(my_cox, [0, 1, 1, 1]) # only the first covariate is tested as 0
        >>> chi-square, pvalue = my_cox.likelihood_ratio_test(my_cox, [0, 0, 0, 1]) # the first three covariates are tested as 0
    """

    if isinstance(c, list):
        assert cox.covar.shape[-1] == len(c)
        c = np.array(c)
    elif isinstance(c, np.ndarray):
        assert len(c.shape) == 1
        assert cox.covar.shape[-1] == c.shape[-1]

    if c is None:
        # null hypothesis is beta = 0
        neg_pl_beta = cox._negative_log_partial_likelihood(cox.param)
        neg_pl_beta_0 = cox._negative_log_partial_likelihood(np.zeros_like(cox.param))
        ch2 = 2 * (neg_pl_beta_0 - neg_pl_beta)
        pval = chi2.sf(ch2, df=cox.covar.shape[-1])
        return round(ch2, 6), round(pval, 6)
    else:
        # local test
        tested_covar = np.where(c != 0)[0]
        other_covar = np.where(c == 0)[0]

        neg_pl_beta = cox._negative_log_partial_likelihood(cox.param)
        cox_under_h0 = Cox(
            cox.time,
            cox.covar[:, tested_covar],
            cox.event,
            cox.entry,
        )
        neg_pl_beta_under_h0 = cox_under_h0._negative_log_partial_likelihood(
            cox_under_h0.fit()
        )
        ch2 = 2 * (neg_pl_beta_under_h0 - neg_pl_beta)
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)


def cox_scores_test(cox: Cox, c: np.ndarray = None) -> Tuple[float, float]:
    """Perform scores test (testing nullity of covariate effect)

    Args:
        cox(Cox): cox object represented a fitted cox model
        c (np.ndarray, optional): combination vector of 0 (beta is 0) and 1 (beta is not 0) indicating which covar coordinate is 0 in the null hypothesis
        Defaults to None, then the null hypothesis corresponds to null effect of all covariates

    Returns:
        Tuple[float, float]: test value and its corresponding pvalue

    Examples:
        >>> my_cox.fit() # beta is of shape (4,)
        >>> chi-square, pvalue = my_cox.scores_test(my_cox, [0, 1, 1, 1]) # only the first covariate is tested as 0
        >>> chi-square, pvalue = my_cox.scores_ratio_test(my_cox, [0, 0, 0, 1]) # the first three covariates are tested as 0
    """

    if isinstance(c, list):
        assert cox.covar.shape[-1] == len(c)
        c = np.array(c)
    elif isinstance(c, np.ndarray):
        assert len(c.shape) == 1
        assert cox.covar.shape[-1] == c.shape[-1]

    if c is None:
        # null hypothesis is beta = 0
        ch2 = np.dot(
            -cox._jac(np.zeros(cox.covar.shape[-1])),
            np.dot(
                linalg.inv(cox._hess(np.zeros(cox.covar.shape[-1]))),
                -cox._jac(np.zeros(cox.covar.shape[-1])),
            ),
        )
        pval = chi2.sf(ch2, df=cox.covar.shape[-1])
        return round(ch2, 3), round(pval, 3)
    else:
        # local test

        tested_covar = np.where(c != 0)[0]
        other_covar = np.where(c == 0)[0]

        cox_under_h0 = Cox(
            cox.time,
            cox.covar[:, tested_covar],
            cox.event,
            cox.entry,
        )
        beta_under_h0 = np.zeros(cox.covar.shape[-1])
        beta_under_h0[tested_covar] = cox_under_h0.fit()

        ch2 = np.dot(
            cox._jac(beta_under_h0)[other_covar],
            np.dot(
                linalg.inv(cox._hess(beta_under_h0))[np.ix_(other_covar, other_covar)],
                cox._jac(beta_under_h0)[other_covar],
            ),
        )
        pval = chi2.sf(ch2, df=len(other_covar))
        return round(ch2, 6), round(pval, 6)
