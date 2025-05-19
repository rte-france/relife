from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife.data import LifetimeData

from ._base import Likelihood

if TYPE_CHECKING:
    from relife.lifetime_model import FittableParametricLifetimeModel


class LikelihoodFromLifetimes(Likelihood):
    """
    Generic likelihood object for parametric_model lifetime model

    Parameters
    ----------
    model : ParametricLifetimeDistribution
        Underlying core used to compute probability functions
    lifetime_data : LifetimeData
        Observed lifetime data used one which the likelihood is evaluated
    """

    def __init__(
        self,
        model: FittableParametricLifetimeModel[*tuple[float | NDArray[np.float64]]],
        lifetime_data: LifetimeData,
    ):
        self.model = copy.deepcopy(model)
        self.data = lifetime_data

    @override
    @property
    def hasjac(self) -> bool:
        return hasattr(self.model, "jac_hf") and hasattr(self.model, "jac_chf")

    def _complete_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.complete is None:
            return None
        return -np.sum(
            np.log(
                self.model.hf(
                    lifetime_data.complete.lifetime_values,
                    *lifetime_data.complete.args,
                )
            )  # (m, 1)
        )  # ()

    def _right_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.complete_or_right_censored is None:
            return None
        return np.sum(
            self.model.chf(
                lifetime_data.complete_or_right_censored.lifetime_values,
                *lifetime_data.complete_or_right_censored.args,
            ),
            dtype=np.float64,  # (m, 1)
        )  # ()

    def _left_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.left_censoring is None:
            return None
        return -np.sum(
            np.log(
                -np.expm1(
                    -self.model.chf(
                        lifetime_data.left_censoring.lifetime_values,
                        *lifetime_data.left_censoring.args,
                    )
                )
            )  # (m, 1)
        )  # ()

    def _left_truncations_contribs(self, lifetime_data: LifetimeData) -> Optional[np.float64]:
        if lifetime_data.left_truncation is None:
            return None
        return -np.sum(
            self.model.chf(
                lifetime_data.left_truncation.lifetime_values,
                *lifetime_data.left_truncation.args,
            ),  # (m, 1)
            dtype=np.float64,
        )  # ()

    def _jac_complete_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.complete is None:
            return None
        return -np.sum(
            self.model.jac_hf(
                lifetime_data.complete.lifetime_values,
                *lifetime_data.complete.args,
                asarray=True,
            )  # (p, m, 1)
            / self.model.hf(
                lifetime_data.complete.lifetime_values,
                *lifetime_data.complete.args,
            ),  # (m, 1)
            axis=(1, 2),
        )  # (p,)

    def _jac_right_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.complete_or_right_censored is None:
            return None
        return np.sum(
            self.model.jac_chf(
                lifetime_data.complete_or_right_censored.lifetime_values,
                *lifetime_data.complete_or_right_censored.args,
                asarray=True,
            ),  # (p, m, 1)
            axis=(1, 2),
        )  # (p,)

    def _jac_left_censored_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.left_censoring is None:
            return None
        return -np.sum(
            self.model.jac_chf(
                lifetime_data.left_censoring.lifetime_values,
                *lifetime_data.left_censoring.args,
                asarray=True,
            )  # (p, m, 1)
            / np.expm1(
                self.model.chf(
                    lifetime_data.left_censoring.lifetime_values,
                    *lifetime_data.left_censoring.args,
                )
            ),  # (m, 1)
            axis=(1, 2),
        )  # (p,)

    def _jac_left_truncations_contribs(self, lifetime_data: LifetimeData) -> Optional[NDArray[np.float64]]:
        if lifetime_data.left_truncation is None:
            return None
        return -np.sum(
            self.model.jac_chf(
                lifetime_data.left_truncation.lifetime_values,
                *lifetime_data.left_truncation.args,
                asarray=True,
            ),  # (p, m, 1)
            axis=(1, 2),
        )  # (p,)

    def negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> np.float64:
        self.model.params = params
        contributions = (
            self._complete_contribs(self.data),
            self._right_censored_contribs(self.data),
            self._left_censored_contribs(self.data),
            self._left_truncations_contribs(self.data),
        )
        return sum(x for x in contributions if x is not None)  # ()

    @override
    def jac_negative_log(
        self,
        params: NDArray[np.float64],  # (p,)
    ) -> NDArray[np.float64]:
        if not self.hasjac:
            raise AttributeError(f"No support of jac negative likelihood for {self.model.__class__.__name__}")
        self.model.params = params
        jac_contributions = (
            self._jac_complete_contribs(self.data),
            self._jac_right_censored_contribs(self.data),
            self._jac_left_censored_contribs(self.data),
            self._jac_left_truncations_contribs(self.data),
        )
        return sum(x for x in jac_contributions if x is not None)  # (p,)
