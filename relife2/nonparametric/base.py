from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from relife2.data import dataclass


def nearest_1dinterp(x: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """Returns x nearest interpolation based on xp and yp data points
    xp has to be monotonically increasing

    Args:
        x (np.ndarray): 1d x coordinates to interpolate
        xp (np.ndarray): 1d known x coordinates
        yp (np.ndarray): 1d known y coordinates

    Returns:
        np.ndarray: interpolation values of x
    """
    spacing = np.diff(xp) / 2
    xp = xp + np.hstack([spacing, spacing[-1]])
    yp = np.concatenate([yp, yp[-1, None]])
    return yp[np.searchsorted(xp, x)]


@dataclass
class Estimates:
    """
    BLABLABLABLA
    """

    timeline: np.ndarray
    values: np.ndarray
    se: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.se is None:
            self.se = np.zeros_like(
                self.values
            )  # garder None/Nan efaire le changement de valeur au niveau du plot

        if self.timeline.shape != self.values.shape != self.se:
            raise ValueError("Incompatible timeline, values and se in Estimates")


class NonParametricLifetimeEstimators(ABC):
    """_summary_"""

    def __init__(
        self,
    ):
        self.estimations = {}

    @abstractmethod
    def estimate(
        self,
        time: ArrayLike,
        event: Optional[ArrayLike] = None,
        entry: Optional[ArrayLike] = None,
        departure: Optional[ArrayLike] = None,
    ) -> Estimates:
        """_summary_

        Returns:
            Tuple[Estimates]: description
        """
