# pyright: basic

from __future__ import annotations

from typing import Any, Generic, Self, TypeVarTuple

import numpy as np

from relife.base import FrozenParametricModel, ParametricModel
from relife.lifetime_model._base import FittableParametricLifetimeModel
from relife.likelihood._base import FittingResults
from relife.stochastic_process._sample import StochasticSampleMapping

Ts = TypeVarTuple("Ts")

__all__ = ["KijimaIProcess", "KijimaIIProcess"]


class KijimaIProcess(ParametricModel, Generic[*Ts]):
    """
    Kijima I Process.
    """

    lifetime_model: FittableParametricLifetimeModel[*Ts]
    fitting_results: FittingResults | None
    q: float | None

    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*Ts], q: float | None = None):
        super().__init__()
        self.lifetime_model = lifetime_model
        self.q = q
        self.fitting_results = None

    def freeze(self, *args: *Ts) -> FrozenKijimaIProcess[*Ts]:
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
        return FrozenKijimaIProcess(self, *args)

    def sample(
        self, nb_samples: int, time_window: tuple[float, float], *args, seed=None
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        This function will sample data and encapsulate them in a StochasticSampleMapping object.

        Parameters
        ----------
        nb_samples : int
            The number of samples
        time_window : tuple of two floats
            Time window in which data are sampled
        *args : float or np.ndarray
            Additional arguments needed by the model.
        seed : int, optional
            Random seed, by default None.

        """
        frozen_kijima = self.freeze(*args)
        return frozen_kijima.sample(nb_samples=nb_samples, time_window=time_window, seed=None)

    def generate_failure_data(
        self
    ) -> dict[str, Any]:
        ...

    def fit(
        self
    ) -> Self:
        ...




class FrozenKijimaIProcess(
    FrozenParametricModel[KijimaIProcess[*Ts], *Ts]
):
    """
    Kijima I process.
    """

    def sample(
        self, nb_samples: int, time_window: tuple[float, float], seed=None
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        nb_samples : int
            The number of samples.
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        seed : int, optional
            Random seed, by default None.

        """

        from relife.utils import get_model_nb_assets
        from ._sample import KijimaIProcessIterable

        iterable = KijimaIProcessIterable(
            self, nb_samples, time_window=time_window, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, get_model_nb_assets(self), nb_samples
        )

    def generate_failure_data(
        self,
    ):
        ...



class KijimaIIProcess(ParametricModel, Generic[*Ts]):
    """
    Kijima II Process.
    """

    lifetime_model: FittableParametricLifetimeModel[*Ts]
    fitting_results: FittingResults | None
    q: float | None

    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*Ts], q: float | None = None):
        super().__init__()
        self.lifetime_model = lifetime_model
        self.q = q
        self.fitting_results = None

    def freeze(self, *args: *Ts) -> FrozenKijimaIIProcess[*Ts]:
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
        return FrozenKijimaIIProcess(self, *args)

    def sample(
        self, nb_samples: int, time_window: tuple[float, float], *args, seed=None
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        This function will sample data and encapsulate them in a StochasticSampleMapping object.

        Parameters
        ----------
        nb_samples : int
            The number of samples
        time_window : tuple of two floats
            Time window in which data are sampled
        *args : float or np.ndarray
            Additional arguments needed by the model.
        seed : int, optional
            Random seed, by default None.

        """
        frozen_kijima = self.freeze(*args)
        return frozen_kijima.sample(nb_samples=nb_samples, time_window=time_window, seed=None)

    def generate_failure_data(
        self
    ) -> dict[str, Any]:
        ...

    def fit(
        self
    ) -> Self:
        ...




class FrozenKijimaIIProcess(
    FrozenParametricModel[KijimaIIProcess[*Ts], *Ts]
):
    """
    Kijima II process.
    """

    def sample(
        self, nb_samples: int, time_window: tuple[float, float], seed=None
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        This function will sample data and encapsulate them in an object.

        Parameters
        ----------
        nb_samples : int
            The number of samples.
        tf : float
            Time at the end of the observation.
        t0 : float, default 0
            Time at the beginning of the observation.
        seed : int, optional
            Random seed, by default None.

        """

        from relife.utils import get_model_nb_assets
        from ._sample import KijimaIIProcessIterable

        iterable = KijimaIIProcessIterable(
            self, nb_samples, time_window=time_window, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, get_model_nb_assets(self), nb_samples
        )

    def generate_failure_data(
        self,
    ):
        ...