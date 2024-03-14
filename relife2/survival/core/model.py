import warnings
from typing import Type

import numpy as np

from ...data.base import databook
from .interface import (
    ParametricFunction,
    ParametricLikelihood,
    ParametricOptimizer,
)
from ..parameter import FittingResult, Parameter


class ParametricModel:
    def __init__(
        self,
        Function: Type[ParametricFunction],
        Likelihood: Type[ParametricLikelihood] = None,
        Optimizer: Type[ParametricOptimizer] = None,
        *params,
        **kparams,
    ):

        if not issubclass(Function, ParametricFunction):
            parent_classes = (Cls.__name__ for Cls in Function.__bases__)
            raise ValueError(
                f"ParametricFunction subclass expected, got '{parent_classes}'"
            )

        if not issubclass(Likelihood, ParametricLikelihood):
            parent_classes = (Cls.__name__ for Cls in Function.__bases__)
            raise ValueError(
                "ParametrictLikelihood subclass expected, got"
                f" '{parent_classes}'"
            )

        if not issubclass(Optimizer, ParametricOptimizer):
            parent_classes = (Cls.__name__ for Cls in Function.__bases__)
            raise ValueError(
                "ParametricOptimizer subclass expected, got"
                f" {Optimizer.__name__} '{parent_classes}'"
            )

        self.function = Function()
        self.Likelihood = Likelihood
        self.Optimizer = Optimizer
        self._init_params_values = self.function.params.values
        self._init_params(*params, **kparams)
        self._fitting_params = Parameter(
            self.function.params.nb_params, self.function.params.param_names
        )
        self._fitting_results = None

    def _init_params(self, *params, **kparams):

        if len(params) != 0 and len(kparams) != 0:
            raise ValueError("Can't specify key word params and params")
        elif len(params) != 0:
            params = np.array(params)
            params_shape = params.shape
            functions_params_shape = self.function.params.values.shape
            if params.shape != self.function.params.values.shape:
                raise ValueError(
                    "Incorrect params shape, expected shape"
                    f" {functions_params_shape} but got {params_shape}"
                )
            self.function.params.values = params
            self._init_params_values = self.function.params.values

        elif len(kparams) != 0:
            missing_param_names = set(
                self.function.params.param_names
            ).difference(set(kparams.keys()))
            if len(missing_param_names) != 0:
                warnings.warn(
                    f"""{missing_param_names} params are missing and will be initialized randomly"""
                )
            for key, value in kparams.items():
                self.function.params[key] = value
            self._init_params_values = self.function.params.values

    def _set_params(self, params):
        if params is None:
            input_params = self._init_params_values
        elif type(params) == list:
            input_params = np.array(params)
        elif type(params) == np.ndarray:
            input_params = params
        elif type(params) == int:
            input_params = np.array(params)
        elif type(params) == float:
            input_params = np.array(params)
        else:
            raise TypeError("Incorrect params type")

        if input_params.shape != self.function.params.values.shape:
            raise ValueError(
                f"""
                expected {self.function.params.values.shape} shape for
                params but got {input_params.shape}
                """
            )
        else:
            self.function.params.values = input_params
        return self.function

    def _get_func(self, func_name: str, *args, **kwargs):
        if hasattr(self.function, func_name):
            self.function = self._set_params(kwargs["params"])
            res = getattr(self.function, func_name)(*args)
        else:
            raise AttributeError(
                "f{self.function.__name__} has no attribute {func_name}"
            )
        self.function.params.values = self._init_params_values
        return res

    def ppf(
        self, probability: np.ndarray, params: np.ndarray = None
    ) -> np.ndarray:
        """_summary_

        Args:
            probability (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("ppf", probability, params=params)

    def median(self, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("median", params=params)

    def rvs(
        self,
        size: int = 1,
        random_state: int = None,
        params: np.ndarray = None,
    ) -> np.ndarray:
        """_summary_

        Args:
            size (int, optional): _description_. Defaults to 1.
            random_state (int, optional): _description_. Defaults to None.
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("rvs", size, random_state, params=params)

    def sf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("sf", time, params=params)

    def cdf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("cdf", time, params=params)

    def pdf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("pdf", time, params=params)

    def isf(
        self, probability: np.ndarray, params: np.ndarray = None
    ) -> np.ndarray:
        """_summary_

        Args:
            probability (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("isf", probability, params=params)

    def hf(self, time: np.ndarray, params: np.ndarray = None):
        """Hazard function

        Args:
            time (np.ndarray): time

        Returns:
            np.ndarray: hazard function values
        """
        return self._get_func("hf", time, params=params)

    def chf(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("chf", time, params=params)

    def mean(self, params: np.ndarray = None) -> float:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            float: _description_
        """
        return self._get_func("mean", params=params)

    def var(self, params: np.ndarray = None) -> float:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            float: _description_
        """
        return self._get_func("var", params=params)

    def mrl(self, time: np.ndarray, params: np.ndarray = None) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("mrl", time, params=params)

    def ichf(
        self, cumulative_hazard_rate: np.ndarray, params: np.ndarray = None
    ) -> np.ndarray:
        """_summary_

        Args:
            cumulative_hazard_rate (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        return self._get_func("ichf", cumulative_hazard_rate, params=params)

    def fit(
        self,
        observed_lifetimes: np.ndarray,
        complete_indicators: np.ndarray = None,
        left_censored_indicators: np.ndarray = None,
        right_censored_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
        **kwargs,
    ):

        if self.Likelihood is None:
            raise ValueError("Model has no Likelihood implemented")

        if self.Optimizer is None:
            raise ValueError("Model has no Optimizer implemented")

        db = databook(
            observed_lifetimes,
            complete_indicators,
            left_censored_indicators,
            right_censored_indicators,
            entry,
            departure,
        )

        likelihood = self.Likelihood(db)
        optimizer = self.Optimizer(likelihood)

        self.function, opt = optimizer.fit(self.function, **kwargs)
        jac = optimizer.likelihood.jac_negative_log_likelihood(self.function)
        var = np.linalg.inv(
            optimizer.likelihood.hess_negative_log_likelihood(
                self.function, **kwargs
            )
        )
        self._fitting_results = FittingResult(
            opt,
            jac,
            var,
            len(db),
        )
        self._fitting_params.values = opt.x

    @property
    def params(self):
        return self.function.params

    @property
    def fitting_results(self):
        if self._fitting_results is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_results

    @property
    def fitting_params(self):
        if self._fitting_params is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_params
