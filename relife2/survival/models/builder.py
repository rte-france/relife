import warnings
from typing import Type, Union

import numpy as np

from .backbone import (
    FittingResults,
    Likelihood,
    Optimizer,
    Parameters,
    ProbabilityFunctions,
)


# builder object
class LifetimeModel:
    def __init__(
        self,
        probability_functions: Type[ProbabilityFunctions],
        likelihood: Type[Likelihood] = None,
        optimizer: Type[Optimizer] = None,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ):

        if not issubclass(probability_functions, ProbabilityFunctions):
            parent_classes = (
                Cls.__name__ for Cls in probability_functions.__bases__
            )
            raise ValueError(
                f"ParametricFunction subclass expected, got '{parent_classes}'"
            )

        if not issubclass(likelihood, Likelihood):
            parent_classes = (Cls.__name__ for Cls in likelihood.__bases__)
            raise ValueError(
                "ParametrictLikelihood subclass expected, got"
                f" '{parent_classes}'"
            )

        if not issubclass(optimizer, Optimizer):
            parent_classes = (Cls.__name__ for Cls in optimizer.__bases__)
            raise ValueError(
                f"Optimizer subclass expected, got {parent_classes}"
            )

        self.pf = probability_functions()
        self.likelihood = likelihood
        self.optimizer = optimizer
        self._set_params(*params, **kwparams)
        self._init_params_values = self.pf.params.values
        self._fitting_params = None
        self._fitting_results = None

    def _set_params(
        self, *params: Union[np.ndarray, float], **kwparams: float
    ) -> None:
        if len(params) != 0 and len(kwparams) != 0:
            raise ValueError("Can't specify key word params and params")

        elif len(params) != 0:
            if len(params) == 1 and type(params[0]) == np.ndarray:
                params = params[0]
            else:
                params = np.array(params)
            params_shape = params.shape
            functions_params_shape = self.pf.params.values.shape
            if params.shape != self.pf.params.values.shape:
                raise ValueError(
                    "Incorrect params shape, expected shape"
                    f" {functions_params_shape} but got {params_shape}"
                )
            self.pf.params.values = params

        elif len(kwparams) != 0:
            missing_param_names = set(self.pf.params.param_names).difference(
                set(kwparams.keys())
            )
            if len(missing_param_names) != 0:
                warnings.warn(
                    f"""{missing_param_names} params are missing and will be initialized randomly"""
                )
            for key, value in kwparams.items():
                self.pf.params[key] = value

    def _get_func(self, func_name: str):
        if hasattr(self.pf, func_name):
            func = getattr(self.pf, func_name)
        else:
            raise AttributeError(
                "f{self.pf.__name__} has no attribute {func_name}"
            )
        return func

    def ppf(
        self,
        probability: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            probability (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("ppf")
        res = func(probability)
        self.pf.params.values = previous_params_values
        return res

    def median(
        self, *params: Union[np.ndarray, float], **kwparams: float
    ) -> np.ndarray:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("median")
        res = func()
        self.pf.params.values = previous_params_values
        return res

    def rvs(
        self,
        size: int = 1,
        random_state: int = None,
        *params: Union[np.ndarray, float],
        **kwparams: float,
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
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("rvs")
        res = func(size, random_state)
        self.pf.params.values = previous_params_values
        return res

    def sf(
        self,
        time: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("sf")
        res = func(time)
        self.pf.params.values = previous_params_values
        return res

    def cdf(
        self,
        time: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("cdf")
        res = func(time)
        self.pf.params.values = previous_params_values
        return res

    def pdf(
        self,
        time: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("pdf")
        res = func(time)
        self.pf.params.values = previous_params_values
        return res

    def isf(
        self,
        probability: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
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
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("isf")
        res = func(probability)
        self.pf.params.values = previous_params_values
        return res

    def hf(
        self,
        time: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """Hazard function

        Args:
            time (np.ndarray): time

        Returns:
            np.ndarray: hazard function values
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("hf")
        res = func(time)
        self.pf.params.values = previous_params_values
        return res

    def chf(
        self,
        time: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("chf")
        res = func(time)
        self.pf.params.values = previous_params_values
        return res

    def mean(
        self, *params: Union[np.ndarray, float], **kwparams: float
    ) -> float:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            float: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("mean")
        res = func()
        self.pf.params.values = previous_params_values
        return res

    def var(
        self, *params: Union[np.ndarray, float], **kwparams: float
    ) -> float:
        """_summary_

        Args:
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            float: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("var")
        res = func()
        self.pf.params.values = previous_params_values
        return res

    def mrl(
        self,
        time: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
    ) -> np.ndarray:
        """_summary_

        Args:
            time (np.ndarray): _description_
            params (np.ndarray, optional): _description_. Defaults to None.

        Raises:
            AttributeError: _description_

        Returns:
            np.ndarray: _description_
        """
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("mrl")
        res = func(time)
        self.pf.params.values = previous_params_values
        return res

    def ichf(
        self,
        cumulative_hazard_rate: np.ndarray,
        *params: Union[np.ndarray, float],
        **kwparams: float,
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
        previous_params_values = self.pf.params.values.copy()
        self._set_params(*params, **kwparams)
        func = self._get_func("ichf")
        res = func(cumulative_hazard_rate)
        self.pf.params.values = previous_params_values
        return res

    def fit(
        self,
        observed_lifetimes: np.ndarray,
        complete_indicators: np.ndarray = None,
        left_censored_indicators: np.ndarray = None,
        right_censored_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
        **kwargs,
    ) -> None:

        if self.likelihood is None:
            raise ValueError("Model has no likelihood implemented")

        if self.optimizer is None:
            raise ValueError("Model has no optimizer implemented")

        likelihood = self.likelihood(
            observed_lifetimes,
            complete_indicators,
            left_censored_indicators,
            right_censored_indicators,
            entry,
            departure,
        )
        optimizer = self.optimizer(self.pf, likelihood)

        self.pf, opt = optimizer.fit(self.pf, likelihood, **kwargs)
        jac = likelihood.jac_negative_log_likelihood(self.pf)
        var = np.linalg.inv(
            likelihood.hess_negative_log_likelihood(self.pf, **kwargs)
        )
        self._fitting_results = FittingResults(
            opt,
            jac,
            var,
            len(observed_lifetimes),
        )

        self._fitting_params = Parameters(
            self.pf.params.nb_params, self.pf.params.param_names
        )
        self._fitting_params.values = opt.x

    @property
    def params(self) -> Parameters:
        return self.pf.params

    @property
    def fitting_results(self) -> FittingResults:
        if self._fitting_results is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_results

    @property
    def fitting_params(self) -> Parameters:
        if self._fitting_params is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method"
                " first",
                UserWarning,
            )
        return self._fitting_params
