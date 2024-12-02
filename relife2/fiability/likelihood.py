from __future__ import annotations

import warnings
from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol, Union

import numpy as np
from numpy.typing import NDArray

from relife2.data import LifetimeData

if TYPE_CHECKING:  # avoid circular imports due to typing
    from relife2.fiability import ParametricLifetimeModel, ParametricModel


class Likelihood(Protocol):
    model: ParametricModel

    @property
    def params(self) -> NDArray[np.float64]:  # read only property allowed with Protocol
        return self.model.params

    @abstractmethod
    def negative_log(self, params: NDArray[np.float64]) -> float:
        """
        Args:
            params ():

        Returns:
            Negative log likelihood value given a set a parameters values
        """


class LikelihoodFromLifetimes(Likelihood):

    def __init__(
        self,
        model: ParametricLifetimeModel[*tuple[NDArray[np.float64], ...]],
        lifetime_data: LifetimeData,
        model_args: tuple[NDArray[np.float64], ...] = (),
    ):
        self.model = model
        self.lifetime_data = lifetime_data
        self.model_args = model_args

    @property
    def hasjac(self):
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
        return (
            self._complete_contribs(self.lifetime_data)
            + self._right_censored_contribs(self.lifetime_data)
            + self._left_censored_contribs(self.lifetime_data)
            + self._left_truncations_contribs(self.lifetime_data)
        )

    def jac_negative_log(
        self,
        params: NDArray[np.float64],
    ) -> Union[None, NDArray[np.float64]]:
        """

        Args:
            params ():

        Returns:

        """
        if not self.hasjac:
            warnings.warn("Model does not support jac negative likelihood natively")
            return None
        self.model.params = params
        return (
            self._jac_complete_contribs(self.lifetime_data)
            + self._jac_right_censored_contribs(self.lifetime_data)
            + self._jac_left_censored_contribs(self.lifetime_data)
            + self._jac_left_truncations_contribs(self.lifetime_data)
        )
