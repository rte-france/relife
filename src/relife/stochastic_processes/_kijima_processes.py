from __future__ import annotations

from typing import Generic, Self, TypeAlias, TypeVarTuple

import numpy as np

from relife.base import FittingResults, ParametricModel
from relife.lifetime_models._base import FittableParametricLifetimeModel

__all__ = [
    "Kijima1Process",
    "Kijima2Process",
    "FrozenKijima1Process",
    "FrozenKijima2Process",
]

Ts = TypeVarTuple("Ts")
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint


class Kijima1Process(ParametricModel, Generic[*Ts]):
    """
    Kijima I Process.
    """

    lifetime_model: FittableParametricLifetimeModel[*Ts]
    fitting_results: FittingResults | None

    def __init__(
        self,
        lifetime_model: FittableParametricLifetimeModel[*Ts],
        q: float | None = None,
    ):
        super().__init__(q=q)
        self.lifetime_model = lifetime_model
        self.fitting_results = None

    @property
    def q(self) -> np.float64:
        return self.get_params()[0]

    def freeze(self, *args: *Ts) -> FrozenKijima1Process[*Ts]:
        """
        Freeze any arguments required by the process into the object data.

        Parameters
        ----------
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenParametricModel
        """
        return FrozenKijima1Process(self, *args)

    def fit(self) -> Self:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Fitting methods for Kijima processes will be introduced in a future release"  # noqa: E501
        )


class FrozenKijima1Process(ParametricModel, Generic[*Ts]):
    """
    Kijima I process.
    """

    unfrozen: Kijima1Process[*Ts]
    args: tuple[*Ts]

    def __init__(
        self,
        kijima_process: Kijima1Process[*Ts],
        *args: *Ts,
    ) -> None:
        super().__init__()
        self.unfrozen = kijima_process
        self.args = args


class Kijima2Process(ParametricModel, Generic[*Ts]):
    """
    Kijima II Process.
    """

    lifetime_model: FittableParametricLifetimeModel[*Ts]
    fitting_results: FittingResults | None

    def __init__(
        self,
        lifetime_model: FittableParametricLifetimeModel[*Ts],
        q: float | None = None,
    ):
        super().__init__(q=q)
        self.lifetime_model = lifetime_model
        self.fitting_results = None

    @property
    def q(self) -> np.float64:
        return self.get_params()[0]

    def freeze(self, *args: *Ts) -> FrozenKijima2Process[*Ts]:
        """
        Freeze any arguments required by the process into the object data.

        Parameters
        ----------
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        FrozenParametricModel
        """
        return FrozenKijima2Process(self, *args)

    def fit(self) -> Self:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Fitting methods for Kijima processes will be introduced in a future release"  # noqa: E501
        )


class FrozenKijima2Process(ParametricModel, Generic[*Ts]):
    """
    Kijima II process.
    """

    unfrozen: Kijima2Process[*Ts]
    args: tuple[*Ts]

    def __init__(
        self,
        kijima_process: Kijima2Process[*Ts],
        *args: *Ts,
    ) -> None:
        super().__init__()
        self.unfrozen = kijima_process
        self.args = args
