from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from scipy.optimize import Bounds

import relife2.distributions as distributions
from relife2 import parametric
from relife2.types import FloatArray


class ShapeFunctions(parametric.Functions, ABC):
    """BLABLABLA"""

    def init_params(self, *args: Any) -> FloatArray:
        return np.ones_like(self.params)

    @property
    def params_bounds(self) -> Bounds:
        """BLABLABLA"""
        return Bounds(
            np.full(self.params.size, np.finfo(float).resolution),
            np.full(self.params.size, np.inf),
        )

    @abstractmethod
    def nu(self, time: FloatArray) -> FloatArray:
        """BLABLABLA"""

    @abstractmethod
    def jac_nu(self, time: FloatArray) -> FloatArray:
        """BLABLABLA"""


class PowerShapeFunctions(ShapeFunctions):
    """BLABLABLA"""

    def __init__(
        self, shape_rate: Optional[float] = None, shape_power: Optional[float] = None
    ):
        super().__init__(shape_rate=shape_rate, shape_power=shape_power)

    def nu(self, time: FloatArray) -> FloatArray:
        return self.shape_rate * time**self.shape_power

    def jac_nu(self, time: FloatArray) -> FloatArray:
        return self.shape_rate * self.shape_power * time ** (self.shape_power - 1)


class ExponentialShapeFunctions(ShapeFunctions):
    """BLABLABLA"""

    def __init__(self, shape_exponent: Optional[float] = None):
        super().__init__(shape_exponent=shape_exponent)

    def nu(self, time: FloatArray) -> FloatArray:
        pass

    def jac_nu(self, time: FloatArray) -> FloatArray:
        pass


# GammaProcessFunctions(FunctionsBridge, parametric.Functions)
class GammaProcessFunctions(parametric.Functions):
    """BLABLABLA"""

    def __init__(
        self,
        shape_function: ShapeFunctions,
        rate: Optional[float] = None,
        initial_resistance: Optional[float] = None,
        load_threshold: Optional[float] = None,
    ):

        super().__init__()
        self.add_functions(
            "process_lifetime_distribution",
            distributions.GPDistributionFunctions(
                shape_function, rate, initial_resistance, load_threshold
            ),
        )

    def init_params(self, *args: Any) -> FloatArray:
        return self.process_lifetime_distribution.init_params(*args)

    @property
    def params_bounds(self) -> Bounds:
        return self.process_lifetime_distribution.params_bounds

    def sample(self):
        """BLABLABLA"""
        pass
