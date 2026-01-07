import numpy as np
from numpy.typing import NDArray

from ._base import Likelihood
from relife.utils import reshape_1d_arg

def _args_reshape(*args):
    args_list = [np.asarray(arg) for arg in args]
    for i, arg in enumerate(args_list):
        if arg.ndim > 2:
            raise ValueError(
                f"Invalid arg shape, got {arg.shape} shape at position {i}"
            )
        if arg.ndim < 2:
            args_list[i] = arg.reshape(-1, 1)
    return tuple(args_list)


class DefaultLifetimeLikelihood(Likelihood):
    def __init__(self, model, time, *args, event = None, entry = None):
        super().__init__(model)
        self.params = self.model._get_initial_params(time, *args, event=event, entry=entry)

        time = reshape_1d_arg(time)
        event = reshape_1d_arg(event) if event is not None else np.ones_like(time, dtype=np.bool_)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time, dtype=np.float64)
        args = _args_reshape(*args)
        sizes = [len(x) for x in (time, event, entry, *args) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {tuple(sizes)}"
            )

        self._time = time
        self._complete_time = time[np.flatnonzero(event)]
        self._nonzero_entry = entry[np.flatnonzero(entry)]
        self._args = args
        self._complete_time_args = tuple(arg[np.flatnonzero(event)] for arg in args)
        self._nonzero_entry_args = tuple(arg[np.flatnonzero(entry)] for arg in args)
        self._nb_observations = len(time)

    @property
    def nb_observations(self):
        return self._nb_observations

    def _time_contrib(self):
        return np.sum(self.model.chf(self._time, *self._args))

    def _event_contrib(self):
        if len(self._complete_time) == 0:
            return None
        return np.sum(-np.log(self.model.hf(self._complete_time, *self._complete_time_args)))

    def _entry_contrib(self):
        if len(self._nonzero_entry) == 0:
            return None
        return -np.sum(self.model.chf(self._nonzero_entry, *self._nonzero_entry_args))

    def _jac_time_contrib(self):
        jac = self.model.jac_chf(
            self._time,
            *self._args,
            asarray=True,
        )

        # Sum all contribs
        # Axis 0 is the parameters
        return np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def _jac_event_contrib(self):
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

    def _jac_entry_contrib(self):
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

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> np.float64:
        self.params = params  # changes model params
        contributions = (
            self._time_contrib(),
            self._event_contrib(),
            self._entry_contrib(),
        )
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
        self.params = params
        jac_contributions = (
            self._jac_time_contrib(),
            self._jac_event_contrib(),
            self._jac_entry_contrib(),
        )
        return sum(x for x in jac_contributions if x is not None)  # (p,)


class IntervalLifetimeLikelihood(Likelihood):
    def __init__(
        self,
        model,
        time_inf: NDArray[np.float64],
        time_sup: NDArray[np.float64],
        *args,
        entry = None,
    ):
        super().__init__(model)
        self.params = self.model._get_initial_params(time_sup, *args, entry=entry)
        time_inf = reshape_1d_arg(time_inf)
        time_sup = reshape_1d_arg(time_sup)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time_inf, dtype=np.float64)
        args = _args_reshape(*args)
        sizes = [len(x) for x in (time_inf, time_sup, entry, *args) if x is not None]
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
        self._censored_time_args = tuple(
            arg[~complete_time_index] for arg in args
        )
        self._nonzero_entry_args = tuple(arg[(entry > 0).squeeze()] for arg in args)

    @property
    def nb_observations(self):
        return self._nb_observations

    def _complete_time_contrib(self):
        if len(self._complete_time == 0):
            return None
        return np.sum(-np.log(self.model.pdf(self._complete_time, *self._complete_time_args)))

    def _interval_censored_time_contrib(self):
        if len(self._censored_time_upper_bound) == 0:
            return None
        return np.sum(
            -np.log(
                10**-10
                + self.model.cdf(self._censored_time_upper_bound, *self._censored_time_args)
                - self.model.cdf(self._censored_time_lower_bound, *self._censored_time_args)
            ),
        )

    def _entry_contrib(self):
        if len(self._nonzero_entry) == 0:
            return None
        return -np.sum(self.model.chf(self._nonzero_entry, *self._nonzero_entry_args))

    def _jac_complete_time_contrib(self):
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


    def _jac_interval_censored_time_contrib(self):
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


    def _jac_entry_contrib(self):
        if len(self._nonzero_entry) == 0:
            return None
        # filter entry==0 to avoid numerical error in jac_chf
        jac = self.model.jac_chf(
            self._nonzero_entry,
            *self._nonzero_entry_args,
            asarray=True,
        )

        return -np.sum(jac, axis=tuple(range(1, jac.ndim)))

    def negative_log(self, params):
        self.params = params
        contributions = (
            self._complete_time_contrib(),
            self._interval_censored_time_contrib(),
            self._entry_contrib(),
        )
        return sum(x for x in contributions if x is not None)  # ()

    def jac_negative_log(self, params):
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
        return sum(x for x in jac_contributions if x is not None)  # (p,)


class PartialLifetimeLikelihood(Likelihood):
    """TODO"""

    def __init__(self, model, time, covar, event = None, entry = None):
        super().__init__(model)
        self.params = self.model._get_initial_params(time, covar, event=event, entry=entry)

        time = reshape_1d_arg(time)
        event = reshape_1d_arg(event) if event is not None else np.ones_like(time, dtype=np.bool_)
        entry = reshape_1d_arg(entry) if entry is not None else np.zeros_like(time, dtype=np.float64)
        sizes = [len(x) for x in (time, event, entry, covar) if x is not None]
        if len(set(sizes)) != 1:
            raise ValueError(
                f"All lifetime data must have the same number of values. Fields length are different. Got {tuple(sizes)}"
            )

        (
            _,    # uncensored sorted untied times
            ordered_event_index,
            self._event_count,
        ) = np.unique(
            time[event == 1],
            return_index=True,
            return_counts=True,
        )
        self._covar = covar
        self._ordered_event_covar = covar[event == 1][ordered_event_index]
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
            self.discount_rates = None
            self.discount_rates_mask = None
        elif method.lower() == "cox":
            self.method = method
            self.s_j = None
            self.discount_rates = None
            self.discount_rates_mask = None
        else:
            raise ValueError(f"method allowed are efron, breslow or cox. Not {method}")