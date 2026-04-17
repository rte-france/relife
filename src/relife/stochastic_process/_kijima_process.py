from __future__ import annotations

from typing import Any, Generic, Self, TypeAlias, TypeVar, TypeVarTuple

import numpy as np
from optype.numpy import Array1D, ArrayND

from relife.base import FittingResults, ParametricModel
from relife.lifetime_model._base import FittableParametricLifetimeModel
from relife.stochastic_process._sample import StochasticSampleMapping

__all__ = ["Kijima1Process", "Kijima2Process"]

Ts = TypeVarTuple("Ts")
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint
M = TypeVar(
    "M",
    bound=FittableParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]],
)


class Kijima1Process(ParametricModel, Generic[M]):
    """
    Kijima I Process.
    """

    lifetime_model: M
    fitting_results: FittingResults | None

    def __init__(self, lifetime_model: M, q: float | None = None):
        super().__init__(q=q)
        self.lifetime_model = lifetime_model
        self.fitting_results = None

    @property
    def q(self) -> np.float64:
        return self.get_params()[0]

    def freeze(self, *args: ST | NumpyST | ArrayND[NumpyST]) -> FrozenKijima1Process[M]:
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

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        *args: ST | NumpyST | ArrayND[NumpyST],
        a0: int | float | Array1D[np.float64] | None = None,
        ar: int | float | Array1D[np.float64] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        Samples data and encapsulates them in a StochasticSampleMapping object.

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
        return frozen_kijima.sample(
            nb_samples=nb_samples, time_window=time_window, a0=a0, ar=ar, seed=seed
        )

    def generate_failure_data(self) -> dict[str, Any]:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Failure data methods for stochastic processes will be introduced in a future release"  # noqa: E501
        )

    def fit(self) -> Self:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Fitting methods for Kijima processes will be introduced in a future release"  # noqa: E501
        )


class FrozenKijima1Process(ParametricModel, Generic[M]):
    """
    Kijima I process.
    """

    unfrozen: Kijima1Process[M]
    args: tuple[ST | NumpyST | ArrayND[NumpyST], ...]

    def __init__(
        self,
        kijima_process: Kijima1Process[M],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> None:
        super().__init__()
        self.unfrozen = kijima_process
        self.args = args

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: int | float | Array1D[np.float64] | None = None,
        ar: int | float | Array1D[np.float64] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        Samples data and encapsulates them in a StochasticSampleMapping object.

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

        from ._sample import Kijima1ProcessIterable

        iterable = Kijima1ProcessIterable(
            self, nb_samples, time_window=time_window, a0=a0, ar=ar, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, iterable.nb_assets, nb_samples
        )

    def generate_failure_data(self) -> dict[str, Any]:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Failure data methods for stochastic processes will be introduced in a future release"  # noqa: E501
        )


class Kijima2Process(ParametricModel, Generic[M]):
    """
    Kijima II Process.
    """

    lifetime_model: M
    fitting_results: FittingResults | None

    def __init__(self, lifetime_model: M, q: float | None = None):
        super().__init__(q=q)
        self.lifetime_model = lifetime_model
        self.fitting_results = None

    @property
    def q(self) -> np.float64:
        return self.get_params()[0]

    def freeze(self, *args: ST | NumpyST | ArrayND[NumpyST]) -> FrozenKijima2Process[M]:
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

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        *args: ST | NumpyST | ArrayND[NumpyST],
        a0: int | float | Array1D[np.float64] | None = None,
        ar: int | float | Array1D[np.float64] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        Samples data and encapsulates them in a StochasticSampleMapping object.

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
        return frozen_kijima.sample(
            nb_samples=nb_samples, time_window=time_window, a0=a0, ar=ar, seed=seed
        )

    def generate_failure_data(self) -> dict[str, Any]:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Failure data methods for stochastic processes will be introduced in a future release"  # noqa: E501
        )

    def fit(self) -> Self:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Fitting methods for Kijima processes will be introduced in a future release"  # noqa: E501
        )


class FrozenKijima2Process(ParametricModel, Generic[M]):
    """
    Kijima II process.
    """

    unfrozen: Kijima2Process[M]
    args: tuple[ST | NumpyST | ArrayND[NumpyST], ...]

    def __init__(
        self,
        kijima_process: Kijima2Process[M],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> None:
        super().__init__()
        self.unfrozen = kijima_process
        self.args = args

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: int | float | Array1D[np.float64] | None = None,
        ar: int | float | Array1D[np.float64] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> StochasticSampleMapping:
        """Renewal data sampling.

        Samples data and encapsulates them in a StochasticSampleMapping object.

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

        from ._sample import Kijima2ProcessIterable

        iterable = Kijima2ProcessIterable(
            self, nb_samples, time_window=time_window, a0=a0, ar=ar, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, iterable.nb_assets, nb_samples
        )

    def generate_failure_data(self) -> dict[str, Any]:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Failure data methods for stochastic processes will be introduced in a future release"  # noqa: E501
        )
