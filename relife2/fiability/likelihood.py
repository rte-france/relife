from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Union

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import approx_fprime
from typing_extensions import override

from relife2.utils.data import LifetimeData

if TYPE_CHECKING:  # avoid circular imports due to typing
    from relife2.fiability import ParametricLifetimeModel, ParametricModel


class Likelihood(Protocol):
    model: ParametricModel

    @property
    def params(self) -> NDArray[np.float64]:  # read only property allowed with Protocol
        return self.model.params

    @property
    def hasjac(self) -> bool:
        return False

    @abstractmethod
    def negative_log(self, params: NDArray[np.float64]) -> float:
        """
        Args:
            params ():

        Returns:
            Negative log likelihood value given a set a parameters values
        """

    def jac_negative_log(
        self,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Args:
            params ():

        Returns:
            Negative log likelihood value given a set a parameters values
        """

        raise NotImplementedError


class LikelihoodFromLifetimes(Likelihood):

    def __init__(
        self,
        model: ParametricLifetimeModel[*tuple[NDArray[np.float64], ...]],
        lifetime_data: LifetimeData,
        model_args: tuple[NDArray[np.float64], ...] = (),
    ):
        self.model = model.copy()  # a copy is made as likelihood modifies model.params
        self.lifetime_data = lifetime_data
        self.model_args = model_args

    @override
    @property
    def hasjac(self) -> bool:
        return hasattr(self.model, "jac_hf") and hasattr(self.model, "jac_chf")

    def _complete_contribs(self, lifetime_data: LifetimeData) -> float:
        return -np.sum(
            np.log(
                self.model.hf(
                    lifetime_data.complete.values,
                    *(args[lifetime_data.complete.index] for args in self.model_args),
                )
            )
        )

    def _right_censored_contribs(self, lifetime_data: LifetimeData) -> float:
        return np.sum(
            self.model.chf(
                lifetime_data.rc.values,
                *(args[lifetime_data.rc.index] for args in self.model_args),
            ),
            dtype=np.float64,
        )

    def _left_censored_contribs(self, lifetime_data: LifetimeData) -> float:
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.model.chf(
                        lifetime_data.left_censored.values,
                        *(
                            args[lifetime_data.left_censored.index]
                            for args in self.model_args
                        ),
                    )
                )
            )
        )

    def _left_truncations_contribs(self, lifetime_data: LifetimeData) -> float:
        return -np.sum(
            self.model.chf(
                lifetime_data.left_truncated.values,
                *(args[lifetime_data.left_truncated.index] for args in self.model_args),
            ),
            dtype=np.float64,
        )

    def _jac_complete_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_hf(
                lifetime_data.complete.values,
                *(args[lifetime_data.complete.index] for args in self.model_args),
            )
            / self.model.hf(
                lifetime_data.complete.values,
                *(args[lifetime_data.complete.index] for args in self.model_args),
            ),
            axis=0,
        )

    def _jac_right_censored_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        return np.sum(
            self.model.jac_chf(
                lifetime_data.rc.values,
                *(args[lifetime_data.rc.index] for args in self.model_args),
            ),
            axis=0,
        )

    def _jac_left_censored_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_chf(
                lifetime_data.left_censored.values,
                *(args[lifetime_data.left_censored.index] for args in self.model_args),
            )
            / np.expm1(
                self.model.chf(
                    lifetime_data.left_censored.values,
                    *(
                        args[lifetime_data.left_censored.index]
                        for args in self.model_args
                    ),
                )
            ),
            axis=0,
        )

    def _jac_left_truncations_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        return -np.sum(
            self.model.jac_chf(
                lifetime_data.left_truncated.values,
                *(args[lifetime_data.left_truncated.index] for args in self.model_args),
            ),
            axis=0,
        )

    def negative_log(
        self,
        params: NDArray[np.float64],
    ) -> float:
        self.model.params = params
        print("complete :", self._complete_contribs(self.lifetime_data))
        print("RC :", self._right_censored_contribs(self.lifetime_data))
        print("LC :", self._left_censored_contribs(self.lifetime_data))
        print("LT :", self._left_truncations_contribs(self.lifetime_data))
        print(
            "total :",
            (
                self._complete_contribs(self.lifetime_data)
                + self._right_censored_contribs(self.lifetime_data)
                + self._left_censored_contribs(self.lifetime_data)
                + self._left_truncations_contribs(self.lifetime_data)
            ),
        )
        return (
            self._complete_contribs(self.lifetime_data)
            + self._right_censored_contribs(self.lifetime_data)
            + self._left_censored_contribs(self.lifetime_data)
            + self._left_truncations_contribs(self.lifetime_data)
        )

    @override
    def jac_negative_log(
        self,
        params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if not self.hasjac:
            raise AttributeError(
                f"No support of jac negative likelihood for {self.model.__class__.__name__}"
            )
        self.model.params = params
        return (
            self._jac_complete_contribs(self.lifetime_data)
            + self._jac_right_censored_contribs(self.lifetime_data)
            + self._jac_left_censored_contribs(self.lifetime_data)
            + self._jac_left_truncations_contribs(self.lifetime_data)
        )


def hessian_cs(
    likelihood: Likelihood,
    eps: float = 1e-6,
) -> Union[NDArray[np.float64], None]:
    """

    Args:
        likelihood ():
        eps ():

    Returns:

    """
    if likelihood.hasjac:
        size = likelihood.params.size
        hess = np.empty((size, size))
        u = eps * 1j * np.eye(size)
        params = likelihood.params.copy()
        for i in range(size):
            for j in range(i, size):
                hess[i, j] = (
                    np.imag(likelihood.jac_negative_log(params + u[i])[j]) / eps
                )
                if i != j:
                    hess[j, i] = hess[i, j]
        return hess
    return None


def hessian_2point(
    likelihood: Likelihood,
    eps: float = 1e-6,
) -> Union[NDArray[np.float64], None]:
    """

    Args:
        likelihood ():
        eps ():

    Returns:

    """
    if likelihood.hasjac:
        size = likelihood.params.size
        params = likelihood.params.copy()
        hess = np.empty((size, size))
        for i in range(size):
            hess[i] = approx_fprime(
                params,
                lambda params: likelihood.jac_negative_log(params)[i],
                eps,
            )
        return hess
    return None


def hessian_from_likelihood(method: str):
    match method:
        case "2-point":
            return hessian_2point
        case "cs":
            return hessian_cs
        case _:
            return hessian_2point
