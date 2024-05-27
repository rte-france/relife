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
            parent_classes = (Cls.__name__ for Cls in probability_functions.__bases__)
            raise ValueError(
                f"ParametricFunction subclass expected, got '{parent_classes}'"
            )

        if not issubclass(likelihood, Likelihood):
            parent_classes = (Cls.__name__ for Cls in likelihood.__bases__)
            raise ValueError(
                "ParametrictLikelihood subclass expected, got" f" '{parent_classes}'"
            )

        if not issubclass(optimizer, Optimizer):
            parent_classes = (Cls.__name__ for Cls in optimizer.__bases__)
            raise ValueError(f"Optimizer subclass expected, got {parent_classes}")

        self.pf = probability_functions()
        self.likelihood = likelihood
        self.optimizer = optimizer
        self._set_params(*params, **kwparams)
        self._init_params_values = self.pf.params.values
        self._fitting_params = None
        self._fitting_results = None

    def _set_params(self, *params: Union[np.ndarray, float], **kwparams: float) -> None:
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
            raise AttributeError("f{self.pf.__name__} has no attribute {func_name}")
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

    def mean(self, *params: Union[np.ndarray, float], **kwparams: float) -> float:
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

    def var(self, *params: Union[np.ndarray, float], **kwparams: float) -> float:
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
        time: np.ndarray,
        lc_indicators: np.ndarray = None,
        rc_indicators: np.ndarray = None,
        entry: np.ndarray = None,
        departure: np.ndarray = None,
        **kwargs,
    ) -> None:

        if self.likelihood is None:
            raise ValueError("Model has no likelihood implemented")

        if self.optimizer is None:
            raise ValueError("Model has no optimizer implemented")

        likelihood = self.likelihood(
            time,
            lc_indicators,
            rc_indicators,
            entry,
            departure,
        )
        optimizer = self.optimizer(self.pf, likelihood)

        self.pf, opt = optimizer.fit(self.pf, likelihood, **kwargs)
        jac = likelihood.jac_negative_log_likelihood(self.pf)
        var = np.linalg.inv(likelihood.hess_negative_log_likelihood(self.pf, **kwargs))
        self._fitting_results = FittingResults(
            opt,
            jac,
            var,
            len(time),
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
                "Model parameters have not been fitted. Call fit() method" " first",
                UserWarning,
            )
        return self._fitting_results

    @property
    def fitting_params(self) -> Parameters:
        if self._fitting_params is None:
            warnings.warn(
                "Model parameters have not been fitted. Call fit() method" " first",
                UserWarning,
            )
        return self._fitting_params


class ParametersSet:
    def __init__(self, *params: Parameters):
        self.params_set = params
        self._nb_params = [p.nb_params for p in params]
        self.nb_params = sum(self._nb_params)
        self._param_names = [p.param_names for p in params]
        self.params_index = {name: i for i, name in enumerate(self.param_names)}

    @property
    def values(self):
        return np.concatenate([p.values for p in self.params_set])

    @property
    def param_names(self):
        return [x for names in self._param_names for x in names]

    @values.setter
    def values(self, v: np.ndarray):
        values_set = np.split(v, np.cumsum(self._nb_params))
        for i, _v in enumerate(values_set):
            self.params_set[i].values = _v

    def __len__(self):
        self.nb_params

    def __getitem__(self, i):
        return self.params_set[self.nb_params // i].values[self.nb_params % i]

    def __setitem__(self, param_name: str, value: float):
        if param_name in self.params_index:
            i = self.params_index[param_name]
            self.params_set[self.nb_params // i].values[self.nb_params % i] = value
        else:
            raise AttributeError(
                f"""
                Parameter has no attr called {param_name}
                """
            )

    def __getattr__(self, attr: str):
        if attr in self.params_index:
            i = self.params_index[attr]
            return self.params_set[self.nb_params // i].values[self.nb_params % i]
        else:
            raise AttributeError(
                f"""
                Parameter has no attr called {attr}
                """
            )

    def __str__(self):
        class_name = type(self).__name__
        res = [f"{name} = {getattr(self, name)} \n" for name in self.param_names]
        res = "".join(res)
        return f"\n{class_name}\n{res}"


class PFComposer(ProbabilityFunctions):
    def __init__(self, *pf: ProbabilityFunctions):
        self.pf_set = pf
        self.params = ParametersSet(*[f.params for f in self.pd_set])
        super().__init__(self.params.nb_params, self.params.param_names)
        # problème : on veut que PFComposer soit un object type ProbabilityFunctions sans avoir à définir les méthodes abstraites
        # idée : __new__
        # Est-ce qu'on ne peut pas avoir une LifetimeModel prenant en entrée plusieurs ProbabilityFunctions objects ?
