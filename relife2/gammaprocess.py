from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from scipy.optimize import Bounds

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

    def nu(self, time: FloatArray) -> FloatArray:
        return self.shape_rate * time**self.shape_power

    def jac_nu(self, time: FloatArray) -> FloatArray:
        return self.shape_rate * self.shape_power * time ** (self.shape_power - 1)


class ExponentialShapeFunction(ShapeFunctions):
    """BLABLABLA"""

    def nu(self, time: FloatArray) -> FloatArray:
        pass

    def jac_nu(self, time: FloatArray) -> FloatArray:
        pass
