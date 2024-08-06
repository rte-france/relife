from typing import Any, Optional, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from relife2.data import Deteriorations, array_factory, deteriorations_factory
from relife2.functions import (
    GPFunctions,
    LikelihoodFromDeteriorations,
    PowerShapeFunctions,
)
from relife2.typing import FloatArray

from .core import ParametricModel


class GammaProcess(ParametricModel):
    """
    BLABLABLABLA
    """

    shape_names: tuple = ("exponential", "power")

    def __init__(
        self,
        shape: str,
        rate: Optional[float] = None,
        **shape_params: Union[float, None],
    ):

        # if shape == "exponential":
        #     shape_functions = ExponentialShapeFunctions(**shape_params)
        if shape == "power":
            shape_functions = PowerShapeFunctions(**shape_params)
        else:
            raise ValueError(
                f"{shape} is not valid name for shape, only {self.shape_names} are allowed"
            )

        super().__init__(GPFunctions(shape_functions, rate))

    def sample(
        self,
        time: ArrayLike,
        unit_ids=ArrayLike,
        nb_sample=1,
        seed=None,
        add_death_time=True,
    ):
        """
        Args:
            time ():
            unit_ids ():
            nb_sample ():
            seed ():
            add_death_time ():

        Returns:

        """
        return self.functions.sample(time, unit_ids, nb_sample, seed, add_death_time)

    def _init_likelihood(
        self,
        deterioration_data: Deteriorations,
        first_increment_uncertainty,
        measurement_tol,
        **kwargs: Any,
    ) -> LikelihoodFromDeteriorations:
        if len(kwargs) != 0:
            extra_args_names = tuple(kwargs.keys())
            raise ValueError(
                f"""
                Distribution likelihood does not expect other data than lifetimes
                Remove {extra_args_names} from kwargs.
                """
            )
        return LikelihoodFromDeteriorations(
            self.functions.copy(),
            deterioration_data,
            first_increment_uncertainty=first_increment_uncertainty,
            measurement_tol=measurement_tol,
        )

    def fit(
        self,
        deterioration_measurements: ArrayLike,
        inspection_times: ArrayLike,
        unit_ids: ArrayLike,
        first_increment_uncertainty: Optional[tuple] = None,
        measurement_tol: np.floating[Any] = np.finfo(float).resolution,
        inplace: bool = True,
        **kwargs: Any,
    ) -> FloatArray:
        """
        BLABLABLABLA
        """

        deterioration_data = deteriorations_factory(
            array_factory(deterioration_measurements),
            array_factory(inspection_times),
            array_factory(unit_ids),
            self.functions.process_lifetime_distribution.initial_resistance,
        )

        param0 = kwargs.pop("x0", self.functions.init_params())

        minimize_kwargs = {
            "method": kwargs.pop("method", "Nelder-Mead"),
            "bounds": kwargs.pop("bounds", self.functions.params_bounds),
            "constraints": kwargs.pop("constraints", ()),
            "tol": kwargs.pop("tol", None),
            "callback": kwargs.pop("callback", None),
            "options": kwargs.pop("options", None),
        }

        likelihood = self._init_likelihood(
            deterioration_data, first_increment_uncertainty, measurement_tol, **kwargs
        )

        optimizer = minimize(
            likelihood.negative_log,
            param0,
            jac=None if not likelihood.hasjac else likelihood.jac_negative_log,
            **minimize_kwargs,
        )

        if inplace:
            self.params = optimizer.x
        return optimizer.x
