from abc import ABC, abstractmethod
from typing import Generic, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife.model import ParametricLifetimeModel

from .hessian_estimation import _hessian_scheme

Args = TypeVarTuple("Args")


class LikelihoodABC(Generic[*Args], ABC):
    model: ParametricLifetimeModel[*Args]

    @property
    def params(self) -> NDArray[np.float64]:
        return self.model.params

    @property
    def hasjac(self) -> bool:
        return False

    def hessian(self, eps: float = 1e-6) -> NDArray[np.float64]:
        return _hessian_scheme(self.model)(self, eps=eps)

    @abstractmethod
    def negative_log(self, params: NDArray[np.float64]) -> float:
        """
        Negative log likelihood.

        Parameters
        ----------
        params : ndarray
            Parameters values on which likelihood is evaluated

        Returns
        -------
        float
            Negative log likelihood value
        """

    def jac_negative_log(
        self,
        params: NDArray[np.float64],
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

        raise NotImplementedError
