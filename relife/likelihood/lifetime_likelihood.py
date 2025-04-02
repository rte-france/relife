from __future__ import annotations

import copy
from typing import TypeVarTuple, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from ._protocol import Likelihood

if TYPE_CHECKING:
    from relife.model import ParametricLifetimeModel
    from .lifetime_data import LifetimeData

Args = TypeVarTuple("Args")


class LikelihoodFromLifetimes(Likelihood[*Args]):
    """
    Generic likelihood object for parametric_model lifetime model

    Parameters
    ----------
    model : ParametricLifetimeDistribution
        Underlying core used to compute probability functions
    lifetime_data : LifetimeData
        Observed lifetime data used one which the likelihood is evaluated
    model_args : tuple of zero or more ndarray, default is ()
        Variadic arguments required by probability functions
    """

    def __init__(
        self,
        model: ParametricLifetimeModel[*Args],
        lifetime_data: LifetimeData,
        model_args: tuple[*Args] = (),
    ):
        self.model = copy.deepcopy(model)
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
                        lifetime_data.left_censoring.values,
                        *(
                            args[lifetime_data.left_censoring.index]
                            for args in self.model_args
                        ),
                    )
                )
            )
        )

    def _left_truncations_contribs(self, lifetime_data: LifetimeData) -> float:
        return -np.sum(
            self.model.chf(
                lifetime_data.left_truncation.values,
                *(
                    args[lifetime_data.left_truncation.index]
                    for args in self.model_args
                ),
            ),
            dtype=np.float64,
        )

    def _jac_complete_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        if hasattr(self.model, "jac_hf"):
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
        raise AttributeError

    def _jac_right_censored_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        if hasattr(self.model, "jac_chf"):
            return np.sum(
                self.model.jac_chf(
                    lifetime_data.rc.values,
                    *(args[lifetime_data.rc.index] for args in self.model_args),
                ),
                axis=0,
            )
        raise AttributeError

    def _jac_left_censored_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        if hasattr(self.model, "jac_chf"):
            return -np.sum(
                self.model.jac_chf(
                    lifetime_data.left_censoring.values,
                    *(
                        args[lifetime_data.left_censoring.index]
                        for args in self.model_args
                    ),
                )
                / np.expm1(
                    self.model.chf(
                        lifetime_data.left_censoring.values,
                        *(
                            args[lifetime_data.left_censoring.index]
                            for args in self.model_args
                        ),
                    )
                ),
                axis=0,
            )
        return AttributeError

    def _jac_left_truncations_contribs(
        self, lifetime_data: LifetimeData
    ) -> NDArray[np.float64]:
        if hasattr(self.model, "jac_chf"):
            return -np.sum(
                self.model.jac_chf(
                    lifetime_data.left_truncation.values,
                    *(
                        args[lifetime_data.left_truncation.index]
                        for args in self.model_args
                    ),
                ),
                axis=0,
            )
        raise AttributeError

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
