import numpy as np
from dataclasses import dataclass

from scipy.optimize import minimize, newton
from scipy.stats import chi2
from scipy import linalg


@dataclass
class Cox:
    r"""Cox regression model

    :math:`h(t,~\vec{z}) = h_0(t)\times e^{\vec{\beta}^\intercal\cdot\vec{z}} = h_0(t) \times g(\vec{z})`

    Only right censored and left truncated times are allowed.
    In case of tied events, two partial likelihood approximations
    are implemented : Breslow and Efron. The baseline is Cox partial likelihood.
    Specific likelihood usage can be set with :code:`.set_method(*arg)`. Otherwise, method is set automatically.

    Attributes:
        time (np.ndarray, shape :math:`n`)
            Age of the assets, :math:`u_i`.
        covar (np.ndarray, shape :math:`(n, p)`)
            Covariates, :math:`\vec{z}_i`. Equivalent of :code:`z_i`.
        event (np.ndarray, shape :math:`n`)
            Type of event, by default None.
        entry (np.ndarray, shape :math:`n`)
            Age of assets at the beginning of the observation period (left truncation), by default None.
        v_j (np.ndarray, shape :math:`m`)
            Ordered distinct ages of the assets, :math:`v_j`.
        event_count (np.ndarray, shape :math:`m`)
            Number of deaths at :math:`v_j`. Equivalent of :code:`d_j`.
        z_j (np.ndarray, shape :math:`(m, p)`)
            Ordered distinct covariates :math:`\vec{z}_j` at :math:`v_j` when they are no ties (:math:`\vec{s}_j` is used for tied events)
        risk_set (np.ndarray, shape :math:`(m, n)`)
            Set of all assets :math:`i` at risk just prior to :math:`v_j`. It corresponds to :math:`\mathbf{R}_j`
        death_set (np.ndarray, shape :math:`(m, n)`)
            Set of all assets :math:`i` who die at :math:`v_j`. Only used for Breslow and Efron method when events are tied. It corresponds to :math:`\mathbf{D}_j`
        s_j (np.ndarray, shape :math:`(m, p)`)
            Sum of ordered distinct covariates :math:`\vec{z}_i` at each :math:`v_j` when events are tied.
        discount_rates (np.ndarray, shape :math:`(m, \text{max}~d_j)`)
            Discount rates applied in Efron method for each set of simultaneous deaths.
        mask_discount_rates (np.ndarray, shape :math:`(m, \text{max}~d_j)`)
            Mask discount rates applied in Efron method for each set of simultaneous deaths.
        method (str, "cox", "breslow" or "efron")
            Method used to compute negative log partial likelihood.
             - If "cox", :math:`\sum_j^{m}\ln\left( \psi_{\mathbf{R}_j}(\vec{z}_i)\right) - \sum_j^{m}\ln(g(\vec{z}_{(j)}))`
             - If "breslow", :math:`\sum_j^{m} d_j \ln\left( \psi\right) - \sum_j^{m}\ln(g(\vec{s}_j))`
             - If "efron", :math:`\sum_j^{m} \sum_{\alpha}^{d_j} \ln\left( \psi_{\mathbf{R}_j} - \frac{\alpha -1}{d_j} \psi_{\mathbf{D}_j}\right) - \sum_j^{m}\ln(g(\vec{s}_j))`
    Notes:
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


    Examples:
        >>> cox = Cox()

    """

    time: np.ndarray
    covar: np.ndarray
    event: np.ndarray = None
    entry: np.ndarray = None

    def __post_init__(self) -> None:
        if self.event is None:
            self.event = np.ones_like(self.time)
        if self.entry is None:
            self.entry = np.zeros_like(self.time)

        assert (
            len(self.time.shape) == 1
            and len(self.event.shape) == 1
            and len(self.covar.shape) == 2
        ), "time and event must be 1d array and covar 2d array"
        assert (
            len(self.time) == len(self.covar) == len(self.event)
        ), "conflicting input data dimensions"

        self._parse_data()

    def _parse_data(
        self,
    ) -> None:
        r"""Parse lifetime data and compute Cox usefull variables

        :code:`v_j`
        :code:`event_count`
        :code:`z_j`
        :code:`risk_set`
        :code:`death_set`
        :code:`s_j`
        :code:`discount_rates`

        Notes
        -----
        Default value for `event` is 1 (no right censoring), default value for `entry` is
        0 (no left truncation).
        """

        if self.event is None:
            self.event = np.ones_like(self.time, int)
        if self.entry is None:
            self.entry = np.zeros_like(self.time, float)

        self.v_j, sorted_uncensored_i, self.event_count = np.unique(
            self.time[self.event == 1],  # uncensored sorted times
            return_index=True,
            return_counts=True,
        )  #: shape :math:`m` - Ordered distinct ages of the assets, :math:`v_j`.

        self.z_j = self.covar[self.event == 1][
            sorted_uncensored_i
        ]  # Y, using sorted_i avoids for loop

        # here risk_set is mask array on time
        # right censored
        # risk_set = np.vstack([time] * len(v_j)) >= np.hstack([v_j[:, None]] * len(time))
        # left truncated & right censored
        self.risk_set = np.logical_and(
            (
                np.vstack([self.entry] * len(self.v_j))
                < np.hstack([self.v_j[:, None]] * len(self.time))
            ),
            (
                np.hstack([self.v_j[:, None]] * len(self.time))
                <= np.vstack([self.time] * len(self.v_j))
            ),
        )

        if (self.event_count > 3).any():
            self.set_method("efron")
        elif (self.event_count <= 3).all() and (2 in self.event_count):
            self.set_method("breslow")
        else:
            self.set_method("cox")

    @property
    def z_i(self) -> np.ndarray:
        """Alias name of :code:`covar`"""

        return self.covar

    @property
    def d_j(self) -> np.ndarray:
        """Alias name of :code:`event_count`"""

        return self.event_count

    @property
    def tied_events(self) -> bool:
        """:Boolean: True if events are tied"""

        return (self.event_count > 1).any()

    def set_method(self, method: str) -> None:
        """Specify method used to compute partial likelihood and its derivates

        Args:
            method (str): "cox", "breslow" or "efron"
        """
        if method.lower() == "efron":
            self.method = method
            self._compute_death_set()
            self._compute_s_j()
            self._compute_efron_discount_rates()
        elif method.lower() == "breslow":
            self.method = method
            self._compute_death_set()
            self._compute_s_j()
            self.discount_rates = None
            self.discount_rates_mask = None
        elif method.lower() == "cox":
            self.method = method
            self.death_set = None
            self.s_j = None
            self.discount_rates = None
            self.discount_rates_mask = None
        else:
            raise ValueError(f"method allowed are efron, breslow or cox. Not {method}")

    @staticmethod
    def _g(z: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """:math:`e^{\vec{\beta}^\intercal \cdot \vec{z}}`"""

        return np.exp(np.dot(z, beta[:, None]))

    @staticmethod
    def _log_g(z: np.ndarray, beta: np.ndarray) -> np.ndarray:
        """math:`\vec{\beta}^\intercal \cdot \vec{z}`"""

        return np.dot(z, beta[:, None])

    def _compute_death_set(self) -> None:
        self.death_set = np.vstack(
            [self.time * self.event] * len(self.v_j)
        ) == np.hstack([self.v_j[:, None]] * len(self.time))

    def _compute_s_j(self) -> None:
        """s_j : [m, p]"""
        self.s_j = np.dot(self.death_set, self.z_i)

    def _compute_efron_discount_rates(self) -> None:
        """
        discount_rates : [m, max(event_count)] or [m, max(event_count)]
        discount_rates_mask : [m, max(event_count)] or [m, max(event_count)]
        """

        self.discount_rates = (
            np.vstack([np.arange(self.d_j.max())] * len(self.d_j)) / self.d_j[:, None]
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
            If order 0, shape m
            If order 1, shape [m, p]
            If order 2, shape [m, p, p]
        """
        if on == "risk":
            i_set = self.risk_set
        elif on == "death":
            i_set = self.death_set

        if order == 0:
            # shape [m]
            return np.dot(i_set, Cox._g(self.z_i, beta))
        elif order == 1:
            # shape [m, p]
            return np.dot(i_set, self.z_i * Cox._g(self.z_i, beta))
        elif order == 2:
            # shape [m, p, p]
            return np.tensordot(
                i_set[:, :None],
                self.z_i[:, None]
                * self.z_i[:, :, None]
                * Cox._g(self.z_i, beta)[:, :, None],
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
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        # neg_L_cox == neg_L_breslow == neg_L_efron if (not self.tied_events)
        if self.method == "cox":
            return -((Cox._log_g(self.z_j, beta)).sum() - np.log(self._psi(beta)).sum())
        elif self.method == "breslow":
            return -(
                (Cox._log_g(self.s_j, beta)).sum()
                - (self.d_j[:, None] * np.log(self._psi(beta))).sum()
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
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        if self.method == "cox":
            return -(
                self.z_j.sum(axis=0)
                - (self._psi(beta, order=1) / self._psi(beta)).sum(axis=0)
            )
        elif self.method == "breslow":
            return -(
                self.s_j.sum(axis=0)
                - (
                    self.d_j[:, None] * (self._psi(beta, order=1) / self._psi(beta))
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
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

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
                return (self.d_j[:, None, None] * hessian_part_1).sum(axis=0) - (
                    self.d_j[:, None, None] * hessian_part_2
                ).sum(axis=0)

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

    def fit(self, use_hessian=True):
        if use_hessian:
            opt = minimize(
                fun=self._negative_log_partial_likelihood,
                x0=np.random.random(self.z_i.shape[1]),
                method="Newton-CG",
                jac=self._jac_negative_log_partial_likelihood,
                hess=self._hess_negative_log_partial_likelihood,
            )
        else:
            opt = minimize(
                fun=self._negative_log_partial_likelihood,
                x0=np.random.random(self.z_i.shape[1]),
                method="L-BFGS-B",
                jac=self._jac_negative_log_partial_likelihood,
            )

        return opt.x

    def chf(self, beta):
        return np.cumsum(self.d_j[:, None] / self._psi(beta))

    def sf(self, beta, covar):
        return np.exp(-self.chf(beta)) ** Cox._g(covar, beta)

    def _cox_snell_residuals(self, beta):
        return self.chf(beta) * np.squeeze(Cox._g(self.z_j, beta))

    def _wald_test(self, beta, beta_0):
        assert beta.shape == beta_0.shape
        information_matrix = self._hess_negative_log_partial_likelihood(beta)
        ch2 = np.dot((beta - beta_0), np.dot(information_matrix, (beta - beta_0)))
        pval = 1 - chi2.cdf(ch2, df=len(beta))
        print(f"chi2 = {ch2} pvalue = {pval}")
        return ch2, pval

    def _likelihood_ratio_test(self, beta, beta_0):
        assert beta.shape == beta_0.shape
        neg_pl_beta = self._negative_log_partial_likelihood(beta)
        neg_pl_beta_0 = self._negative_log_partial_likelihood(beta_0)
        ch2 = 2 * (neg_pl_beta_0 - neg_pl_beta)
        pval = 1 - chi2.cdf(ch2, df=len(beta))
        print(f"chi2 = {ch2} pvalue = {pval}")
        return ch2, pval

    def _scores_test(self, beta, beta_0):
        assert beta.shape == beta_0.shape
        information_matrix = self._hess_negative_log_partial_likelihood(beta_0)
        inverse_information_matrix = linalg.inv(information_matrix)
        jac = -self._jac_negative_log_partial_likelihood(beta_0)
        ch2 = np.dot(jac, np.dot(inverse_information_matrix, jac))
        pval = 1 - chi2.cdf(ch2, df=len(beta))
        print(f"chi2 = {ch2} pvalue = {pval}")
        return ch2, pval
