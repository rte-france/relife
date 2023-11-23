import numpy as np
from dataclasses import dataclass

from scipy.optimize import minimize, newton
from scipy.stats import chi2
from scipy import linalg


@dataclass
class Cox:
    """Cox regression model

    time : [n]
    covar, z_i : [n, p]
    event : [n]
    entry : [n]

    m : nb of uncensored data
    """

    time: np.ndarray
    covar: np.ndarray
    event: np.ndarray
    entry: np.ndarray

    def __post_init__(self):
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

        self.parse_data()

    def parse_data(
        self,
    ):
        """
        v_j : [m]
        event_count, d_j : [m]
        z_j : [m, p]
        risk_set : [m, n] -> nb of alive at v_j
        death_set : [m, n] -> i who died at v_j
        s_j : [m, p]
        discount_rates : [m, max(event_count)] or [m, max(event_count)]
        """
        # sorted_i = np.argsort(self.time)

        self.v_j, sorted_uncensored_i, self.event_count = np.unique(
            self.time[self.event == 1],  # uncensored sorted times
            return_index=True,
            return_counts=True,
        )  # T, N

        self.z_j = self.covar[self.event == 1][
            sorted_uncensored_i
        ]  # Y, using sorted_i avoids for loop

        # here risk_set is mask array on time
        # right censored
        # risk_set = np.vstack([time] * len(v_j)) >= np.hstack([v_j[:, None]] * len(time))
        # left truncated & right censored
        self.risk_set = (
            np.vstack([self.entry] * len(self.v_j))
            <= np.hstack([self.v_j[:, None]] * len(self.time))
        ) & (
            np.hstack([self.v_j[:, None]] * len(self.time))
            <= np.vstack([self.time] * len(self.v_j))
        )

        self.death_set = np.vstack([self.time] * len(self.v_j)) == np.hstack(
            [self.v_j[:, None]] * len(self.time)
        )

        self.s_j = np.dot(self.death_set, self.z_i)

        # if self.Efron_approx:
        # discount_rates is like a mask array of shape [d, max(d_j)]
        self.discount_rates = (
            np.vstack([np.arange(self.event_count.max())] * len(self.d_j))
            / self.event_count[:, None]
        )
        self.discount_rates_mask = np.where(self.discount_rates < 1, 1, 0)

    @property
    def z_i(self):
        """alias name of covar"""

        return self.covar

    @property
    def d_j(self):
        """alias name of event_count"""

        return self.event_count

    @property
    def tied_events(self):
        """are some events tied ?"""

        return (self.event_count > 1).any()

    @property
    def method(self):
        # if True use Efron approximation for likelihood, otherwise, Breslow
        if (self.event_count > 3).any():
            return "efron"
        elif (self.event_count > 1 and self.event_count <= 3).any():
            return "breslow"
        else:
            "cox"

    @staticmethod
    def _g(z, beta):
        """e^{\vec{\beta}^\intercal \cdot \vec{z}}"""

        return np.exp(np.dot(z, beta[:, None]))

    def _psi(self, beta, on="risk", order=0):
        """
        order 0 : [m]
        order 1 : [m, p]
        order 2 : [m, p, p]
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

    def _psi_efron(self, beta, order=0):
        """
        psi for efron formula
        the sum on alpha implies one more dimension

        order 0 : [m, max(d_j)]
        order 1 : [m, max(d_j), p]
        order 2 : [m, max(d_j), p, p]

        discount_rates : [m, max(d_j)]
        discount_rates_mask : [m, max(d_j)]
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

    def _negative_log_partial_likelihood(self, beta, method="cox"):
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        # neg_L_cox == neg_L_breslow == neg_L_efron if (not self.tied_events)
        if method == "cox":
            return -(
                np.log(Cox._g(self.z_j, beta)).sum() - np.log(self._psi(beta)).sum()
            )
        elif method == "breslow":
            return -(
                np.log(Cox._g(self.s_j, beta)).sum()
                - (self.d_j[:, None] * np.log(self._psi(beta))).sum()
            )
        elif method == "efron":
            # .sum(axis=1, keepdims=True) --> sum on alpha to d_j
            # .sum() --> sum on j
            # using where in np.log allows to avoid 0. masked elements
            m = self._psi_efron(beta)
            neg_L_efron = -(
                np.log(Cox._g(self.z_j, beta)).sum()
                - np.log(m, out=np.zeros_like(m), where=(m != 0))
                .sum(axis=1, keepdims=True)
                .sum()
            )
            return neg_L_efron

    def _jac_negative_log_partial_likelihood(self, beta, method="cox"):
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        if method == "cox":
            return -(
                self.z_j.sum(axis=0)
                - (self._psi(beta, order=1) / self._psi(beta)).sum(axis=0)
            )
        elif method == "breslow":
            return -(
                self.s_j.sum(axis=0)
                - (
                    self.d_j[:, None] * (self._psi(beta, order=1) / self._psi(beta))
                ).sum(axis=0)
            )
        elif method == "efron":
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

    def _hess_negative_log_partial_likelihood(self, beta, method="cox"):
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        if method == "cox" or method == "breslow":
            psi_order_0 = self._psi(beta)
            psi_order_1 = self._psi(beta, order=1)

            hessian_part_1 = self._psi(beta, order=2) / psi_order_0[:, :, None]
            # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

            hessian_part_2 = (psi_order_1 / psi_order_0)[:, None] * (
                psi_order_1 / psi_order_0
            )[:, :, None]
            # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

            if method == "cox":
                return hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)
            elif method == "breslow":
                return (self.d_j[:, None, None] * hessian_part_1).sum(axis=0) - (
                    self.d_j[:, None, None] * hessian_part_2
                ).sum(axis=0)

        elif method == "efron":
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

    # def fit_newton(self):
    #     return newton(
    #         func=self._negative_log_partial_likelihood,
    #         x0=np.random.random(self.z_i.shape[1]),
    #         fprime=self._jac_negative_log_partial_likelihood,
    #         fprime2=self._hess_negative_log_partial_likelihood,
    #     )

    # def _wald_test(self, beta, beta_0):
    #     assert beta.shape == beta_0.shape
    #     information_matrix = self._hess_negative_log_partial_likelihood(beta)
    #     ch2 = np.dot((beta - beta_0), np.dot(information_matrix, (beta - beta_0)))
    #     pval = 1 - chi2.cdf(ch2, df=len(beta))
    #     print(f"chi2 = {ch2} pvalue = {pval}")
    #     return ch2, pval

    # def _likelihood_ratio_test(self, beta, beta_0):
    #     assert beta.shape == beta_0.shape
    #     neg_pl_beta = self._negative_log_partial_likelihood(beta)
    #     neg_pl_beta_0 = self._negative_log_partial_likelihood(beta_0)
    #     ch2 = 2 * (neg_pl_beta_0 - neg_pl_beta)
    #     pval = 1 - chi2.cdf(ch2, df=len(beta))
    #     print(f"chi2 = {ch2} pvalue = {pval}")
    #     return ch2, pval

    # def _scores_test(self, beta, beta_0):
    #     assert beta.shape == beta_0.shape
    #     information_matrix = self._hess_negative_log_partial_likelihood(beta_0)
    #     inverse_information_matrix = linalg.inv(information_matrix)
    #     jac = -self._jac_negative_log_partial_likelihood(beta_0)
    #     ch2 = np.dot(jac, np.dot(inverse_information_matrix, jac))
    #     pval = 1 - chi2.cdf(ch2, df=len(beta))
    #     print(f"chi2 = {ch2} pvalue = {pval}")
    #     return ch2, pval
