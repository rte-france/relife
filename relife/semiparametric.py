import numpy as np
from dataclasses import dataclass

from scipy.optimize import minimize, newton
from scipy.stats import chi2
from scipy import linalg


@dataclass
class Cox:
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
        # sorted_i = np.argsort(self.time)

        self.v_j, sorted_uncensored_i, self.event_count = np.unique(
            self.time[self.event == 1],  # uncensored sorted times
            return_index=True,
            return_counts=True,
        )  # T, N

        # if True use Efron approximation for likelihood, otherwise, Breslow
        self.Efron_approx = (self.event_count > 3).any()

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

        self.death_set = (
            np.vstack([self.time] * len(self.v_j)) 
            == np.hstack([self.v_j[:, None]] * len(self.time))
        )

        self.s_j = np.dot(self.death_set, self.z_i)

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
    def is_Efron_used(self):

        return self.Efron_approx

    @property
    def is_Breslow_used(self):
        """are some events tied ?"""

        return not self.Efron_approx


    @staticmethod
    def _g(z, beta):
        """e^{\vec{\beta}^\intercal \cdot \vec{z}}"""

        return np.exp(np.dot(z, beta[:, None]))

    def _risk_gz_sum(self, beta):
        """\sum_{i\in \mathbf{R}(v_j)} g(\vec{z}_i)
        
        or

        \psi_{\mathbf{R}_j}(\vec{z}_i)
        """

        if (not self.tied_events) or self.is_Breslow_used :

            risk_gz_sum = np.dot(self.risk_set, Cox._g(self.z_i, beta))
            # print("risk_gz_sum [d, 1]:", risk_gz_sum.shape)
            return risk_gz_sum
        
        else:

            # psi for Efron
            pass

    def _risk_covar_gz_sum(self, beta):
        """
        \sum_{i\in\mathbf{R}(v_j)} z_{ik} \cdot g(\vec{z}_i)
        
        or 


        \psi_{\mathbf{R}_j}(k,~\vec{z}_i)
        """

        if (not self.tied_events) or self.is_Breslow_used :

            risk_covar_gz_sum = np.dot(self.risk_set, self.z_i * Cox._g(self.z_i, beta))
            # print("risk_covar_gz_sum [d, p]:", risk_covar_gz_sum.shape)
            return risk_covar_gz_sum

        else:

            # psi for Efron
            pass

    def _risk_covar_matrix_gz_sum(self, beta):
        """
        \sum_{i\in\mathbf{R}(v_j)} z_{ik} \cdot z_{ih} \cdot g(\vec{z}_i)

        or

        \psi_{\mathbf{R}_j}(h,~k,~\vec{z}_i)
        """

        if (not self.tied_events) or self.is_Breslow_used :
            risk_covar_matrix_gz_sum = np.tensordot(
                self.risk_set[:, :None],
                self.z_i[:, None]
                * self.z_i[:, :, None]
                * Cox._g(self.z_i, beta)[:, :, None],
                axes=1,
            )
            # print("risk_covar_matrix_gz_sum [d, p, p]:", risk_covar_matrix_gz_sum.shape)
            return risk_covar_matrix_gz_sum
        
        else:

            # psi for Efron
            pass

    def _negative_log_partial_likelihood(self, beta):
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        return -(
            np.log(Cox._g(self.z_j, beta)).sum() - np.log(self._risk_gz_sum(beta)).sum()
        )

    def _jac_negative_log_partial_likelihood(self, beta):
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        return -(
            self.z_j.sum(axis=0)
            - (self._risk_covar_gz_sum(beta) / self._risk_gz_sum(beta)).sum(axis=0)
        )

    def _hess_negative_log_partial_likelihood(self, beta):
        assert len(beta.shape) == 1, "beta must be 1d array"
        assert len(beta) == self.z_i.shape[1], "conflicting beta dimension with covar"

        risk_gz_sum = self._risk_gz_sum(beta)
        risk_covar_gz_sum = self._risk_covar_gz_sum(beta)

        hessian_part_1 = self._risk_covar_matrix_gz_sum(beta) / risk_gz_sum[:, :, None]
        # print("hessian_part_1 [d, p, p]:", hessian_part_1.shape)

        hessian_part_2 = (risk_covar_gz_sum / risk_gz_sum)[:, None] * (
            risk_covar_gz_sum / risk_gz_sum
        )[:, :, None]
        # print("hessian_part_2 [d, p, p]:", hessian_part_2.shape)

        hessian = hessian_part_1.sum(axis=0) - hessian_part_2.sum(axis=0)
        # print("hessian [p, p]:", hessian.shape)

        return hessian
    
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
