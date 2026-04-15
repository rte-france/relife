from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import field
from typing import Any, Generic, Self, TypeAlias, TypeVar, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array1D, Array2D, ArrayND
from typing_extensions import TypeIs

from relife.base import FittingResults, ParametricModel
from relife.lifetime_model._base import (
    FittableParametricLifetimeModel,
    FrozenParametricLifetimeModel,
    LifetimeLikelihood,
)
from relife.stochastic_process._sample import StochasticSampleMapping

__all__ = [
    "NonHomogeneousPoissonProcess",
    "FrozenNonHomogeneousPoissonProcess",
    "is_non_homogeneous_poisson_process",
]

Ts = TypeVarTuple("Ts")
ST: TypeAlias = int | float
NumpyST: TypeAlias = np.floating | np.uint
M = TypeVar(
    "M",
    bound=FittableParametricLifetimeModel[*tuple[ST | NumpyST | ArrayND[NumpyST], ...]],
)


class NonHomogeneousPoissonProcess(ParametricModel, Generic[M]):
    """
    Non-homogeneous Poisson process.
    """

    lifetime_model: M
    fitting_results: FittingResults | None

    def __init__(self, lifetime_model: M):
        super().__init__()
        self.lifetime_model = lifetime_model
        self.fitting_results = None

    def intensity(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        """
        The intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.hf(time, *args)

    def cumulative_intensity(
        self,
        time: ST | NumpyST | ArrayND[NumpyST],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ) -> np.float64 | ArrayND[np.float64]:
        """
        The cumulative intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.lifetime_model.chf(time, *args)

    def freeze(
        self, *args: ST | NumpyST | ArrayND[NumpyST]
    ) -> FrozenNonHomogeneousPoissonProcess[M]:
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
        return FrozenNonHomogeneousPoissonProcess(self, *args)

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        *args: ST | NumpyST | ArrayND[NumpyST],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
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

        from ._sample import NonHomogeneousPoissonProcessIterable

        frozen_nhpp = self.freeze(*args)
        iterable = NonHomogeneousPoissonProcessIterable(
            frozen_nhpp, nb_samples, time_window=time_window, a0=a0, ar=ar, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, iterable.nb_assets, nb_samples
        )

    def generate_failure_data(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        *args: ST | NumpyST | ArrayND[NumpyST],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> dict[str, Any]:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Failure data methods for stochastic processes will be introduced in a future release"  # noqa: E501
        )

    def fit(
        self,
        ages_at_events: Array1D[np.float64],
        events_assets_ids: Sequence[str],
        first_ages: Array1D[np.float64] | None = None,
        last_ages: Array1D[np.float64] | None = None,
        lifetime_model_args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        assets_ids: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Self:
        """
        Estimation of the process parameters from recurrent failure data.

        Parameters
        ----------
        ages_at_events : 1D array of float
            Array of float containing the ages of each asset when the events occured
        events_assets_ids : sequence of hashable
            Sequence object containing the ids of each assets corresponding to
            each ages.
        first_ages : 1D array of float, optional
            Array of float containing the ages of each asset before observing
            events. If set, ``assets_ids`` is needed and its length
            must equal the size of ``first_ages``.
        last_ages : 1D array of float, optional
            Array of float containing the ages of each asset at the end of the
            observation period. If set, ``assets_ids`` is needed and its length
            must equal the size of ``last_ages``.
        lifetime_model_args : tuple of np.ndarray, optional
            Additional arguments needed by the model. If set, ``assets_ids`` is
            needed. For 1D array, the size must equal the length of
            ``assets_ids``. For 2D array (e.g. covar of regression), the length
            of first axis must equal the length of ``assets_ids``.
        assets_ids : sequence of hashable, optional
            Only needed if either ``first_ages``, ``last_ages`` or
            ``model_args`` is filled. It must be a sequence object containing
            the unique ids corresponding to each values contained in
            ``first_ages``, ``last_ages`` and/or ``model_args``

        Returns
        -------
        Self
            The current object with the estimated parameters setted inplace.

        Examples
        --------

        Ages of assets AB2 and CX13 at each event.

        >>> from relife.lifetime_model import Weibull
        >>> from relife.stochastic_process import NonHomogeneousPoissonProcess
        >>> nhpp = NonHomogeneousPoissonProcess(Weibull())
        >>> nhpp.fit(
            np.array([11., 13., 21., 25., 27.]),
            ("AB2", "CX13", "AB2", "AB2", "CX13"),
        )

        With additional information and model args (regression of 2 coefficients)

        >>> from relife.lifetime_model import ParametricProportionalHazard
        >>> nhpp = NonHomogeneousPoissonProcess(ParametricProportionalHazard())
        >>> nhpp.fit(
            np.array([11., 13., 21., 25., 27.]),
            ("AB2", "CX13", "AB2", "AB2", "CX13"),
            first_ages = np.array([10., 12.]),
            last_ages = np.array([35., 60.]),
            model_args = (np.array([[1.2, 5.5], [37.2, 22.2]]),) # 2d array of 2 raws (2 assets) and 2 columns (2 coefficients)
        )
        """  # noqa: E501
        warnings.warn(  # noqa: B028
            "Fit method of NHPP will change in a future release", DeprecationWarning
        )

        nhpp_data = NHPPData(
            ages_at_events,
            events_assets_ids,
            first_ages=first_ages,
            last_ages=last_ages,
            model_args=lifetime_model_args,
            assets_ids=assets_ids,
        )
        time, event, entry, args = nhpp_data.to_lifetime_data()
        optimizer: LifetimeLikelihood[
            FittableParametricLifetimeModel[*tuple[Any, ...]]
        ] = self.lifetime_model.init_likelihood(time, args, event, entry, **kwargs)
        fitting_results = optimizer.optimize()
        self.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(ParametricModel, Generic[M]):
    """
    Non-homogeneous Poisson process.
    """

    unfrozen: NonHomogeneousPoissonProcess[M]
    args: tuple[ST | NumpyST | ArrayND[NumpyST], ...]

    def __init__(
        self,
        nhpp: NonHomogeneousPoissonProcess[M],
        *args: ST | NumpyST | ArrayND[NumpyST],
    ):
        super().__init__()
        self.unfrozen = nhpp
        self.args = args

    def intensity(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        """
        The intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen.intensity(time, *self.args)

    def cumulative_intensity(
        self, time: ST | NumpyST | ArrayND[NumpyST]
    ) -> np.float64 | ArrayND[np.float64]:
        """
        The cumulative intensity function of the process.

        Parameters
        ----------
        time : float or np.ndarray
            Elapsed time value(s) at which to compute the function.
            If ndarray, allowed shapes are ``()``, ``(n,)`` or ``(m, n)``.
        *args : float or np.ndarray
            Additional arguments needed by the model.

        Returns
        -------
        np.float64 or np.ndarray
            Function values at each given time(s).
        """
        return self.unfrozen.cumulative_intensity(time, *self.args)

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: ST | NumpyST | Array1D[NumpyST] | None = None,
        ar: ST | NumpyST | Array1D[NumpyST] | None = None,
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
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
        return self.unfrozen.sample(
            nb_samples, time_window, *self.args, a0=a0, ar=ar, seed=seed
        )

    def generate_failure_data(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        seed: int
        | np.random.Generator
        | np.random.BitGenerator
        | np.random.RandomState
        | None = None,
    ) -> dict[str, Any]:
        """Generates failure data.

        Generates failure data that can be used to fit a non-homogeneous Poisson process.

        Parameters
        ----------
        nb_samples : int
            The number of samples.
        time_window : tuple of two floats
            Time window in which data are sampled
        seed : int, optional
            Random seed, by default None.

        Returns
        -------
        A dict of ages_at_events, events_assets_ids, first_ages, last_ages, model_args and assets_ids
        """  # noqa: E501
        return self.unfrozen.generate_failure_data(
            nb_samples, time_window, *self.args, seed=seed
        )


# typeguard function
def is_non_homogeneous_poisson_process(
    model: NonHomogeneousPoissonProcess[M] | FrozenParametricLifetimeModel[M],
) -> TypeIs[NonHomogeneousPoissonProcess[M] | FrozenParametricLifetimeModel[M]]:
    """
    Checks if model is a non-homogeneous Poisson process.
    """

    return isinstance(
        model, (NonHomogeneousPoissonProcess, FrozenNonHomogeneousPoissonProcess)
    )


class NHPPData:
    ages_at_events: Array1D[np.float64]
    events_assets_ids: Array1D[np.uint32]
    first_ages: Array1D[np.float64] | None
    last_ages: Array1D[np.float64] | None
    model_args: (
        Array1D[Any] | Array2D[Any] | tuple[Array1D[Any] | Array2D[Any], ...] | None
    )
    assets_ids: Array1D[np.uint32] | None

    first_age_index: NDArray[np.int64] = field(repr=False, init=False)
    last_age_index: NDArray[np.int64] = field(repr=False, init=False)

    def __init__(
        self,
        ages_at_events: Array1D[np.float64],
        events_assets_ids: Sequence[str],
        first_ages: Array1D[np.float64] | None = None,
        last_ages: Array1D[np.float64] | None = None,
        model_args: Array1D[Any]
        | Array2D[Any]
        | tuple[Array1D[Any] | Array2D[Any], ...]
        | None = None,
        assets_ids: Sequence[str] | None = None,
    ) -> None:

        # convert inputs to arrays
        self.ages_at_events = np.asarray(ages_at_events, dtype=np.float64)
        self.events_assets_ids = np.unique(
            np.asarray(events_assets_ids), return_inverse=True
        )[1].astype(np.uint32)
        if assets_ids is not None:
            self.assets_ids = np.unique(np.asarray(assets_ids), return_inverse=True)[
                1
            ].astype(np.uint32)
        if first_ages is not None:
            self.first_ages = np.asarray(first_ages, dtype=np.float64)
        if last_ages is not None:
            self.last_ages = np.asarray(last_ages, dtype=np.float64)
        self.model_args = model_args
        self._sanity_checks()

        # sort fields
        sort_ind = np.lexsort((self.ages_at_events, self.events_assets_ids))
        self.events_assets_ids = self.events_assets_ids[sort_ind]
        self.ages_at_events = self.ages_at_events[sort_ind]

        # number of age value per asset id
        nb_ages_per_asset = np.unique_counts(self.events_assets_ids).counts
        # index of the first ages and last ages in ages
        self.first_age_index = np.where(
            np.roll(self.events_assets_ids, 1) != self.events_assets_ids
        )[0]
        self.last_age_index = np.append(
            self.first_age_index[1:] - 1, len(self.events_assets_ids) - 1
        )

        if self.assets_ids is not None:
            # sort fields
            sort_ind = np.argsort(self.assets_ids)
            self.assets_ids = self.assets_ids[sort_ind]
            self.first_ages = (
                self.first_ages[sort_ind]
                if self.first_ages is not None
                else self.first_ages
            )
            self.last_ages = (
                self.last_ages[sort_ind]
                if self.last_ages is not None
                else self.last_ages
            )
            self.model_args = (
                tuple(arg[sort_ind] for arg in self.model_args)
                if self.model_args is not None
                else self.model_args
            )

            if self.first_ages is not None and np.any(
                self.ages_at_events[self.first_age_index]
                <= self.first_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each first_ages value must be lower than all of its corresponding ages values"  # noqa: E501
                )
            if self.last_ages is not None and np.any(
                self.ages_at_events[self.last_age_index]
                >= self.last_ages[nb_ages_per_asset != 0]
            ):
                raise ValueError(
                    "Each last_ages value must be greater than all of its corresponding ages values"  # noqa: E501
                )

    def _sanity_checks(self) -> None:
        # control shapes
        if self.events_assets_ids.ndim != 1:
            raise ValueError(
                "Invalid array shape for events_assets_ids. Expected 1d-array"
            )
        if self.ages_at_events.ndim != 1:
            raise ValueError("Invalid array shape for ages. Expected 1d-array")
        if len(self.events_assets_ids) != len(self.ages_at_events):
            raise ValueError(
                "Shape of events_assets_ids and ages must be equal. Expected equal length 1d-arrays"  # noqa: E501
            )
        if self.assets_ids is not None:
            if self.assets_ids.ndim != 1:
                raise ValueError(
                    "Invalid array shape for assets_ids. Expected 1d-array"
                )
            if self.first_ages is not None:
                if self.first_ages.ndim != 1:
                    raise ValueError(
                        "Invalid array shape for start_ages. Expected 1d-array"
                    )
                if len(self.first_ages) != len(self.assets_ids):
                    raise ValueError(
                        "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"  # noqa: E501
                    )
            if self.last_ages is not None:
                if self.last_ages.ndim != 1:
                    raise ValueError(
                        "Invalid array shape for last_ages. Expected 1d-array"
                    )
                if len(self.last_ages) != len(self.assets_ids):
                    raise ValueError(
                        "Shape of assets_ids and last_ages must be equal. Expected equal length 1d-arrays"  # noqa: E501
                    )
            if bool(self.model_args):
                for arg in self.model_args:
                    arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                    if arg.ndim > 2:
                        raise ValueError(
                            "Invalid arg shape in model_args. Arrays must be 0, 1 or 2d"
                        )
                    try:
                        _ = arg.reshape((len(self.assets_ids), -1))
                    except ValueError as err:
                        raise ValueError(
                            """
                            Invalid arg shape in model_args. Arrays must
                            coherent with the number of assets given by
                            assets_ids
                            """
                        ) from err
        else:
            if self.first_ages is not None:
                raise ValueError(
                    "If first_ages is given, corresponding asset ids must be given in assets_ids"  # noqa: E501
                )
            if self.last_ages is not None:
                raise ValueError(
                    "If last_ages is given, corresponding asset ids must be given in assets_ids"  # noqa: E501
                )
            if bool(self.model_args):
                raise ValueError(
                    "If model_args is given, corresponding asset ids must be given in assets_ids"  # noqa: E501
                )

    def to_lifetime_data(
        self,
    ) -> tuple[
        Array1D[np.float64],
        Array1D[np.bool_],
        Array1D[np.float64],
        Array1D[Any] | Array2D[Any] | tuple[Array1D[Any] | Array2D[Any], ...] | None,
    ]:
        event = np.ones_like(self.ages_at_events, dtype=np.bool_)
        # insert_index = np.cumsum(nb_ages_per_asset)
        # insert_index = last_age_index + 1
        if self.last_ages is not None:
            time = np.insert(
                self.ages_at_events, self.last_age_index + 1, self.last_ages
            )
            event = np.insert(event, self.last_age_index + 1, False)
            _ids = np.insert(
                self.events_assets_ids, self.last_age_index + 1, self.assets_ids
            )
            if self.first_ages is not None:
                entry = np.insert(
                    self.ages_at_events,
                    np.insert((self.last_age_index + 1)[:-1], 0, 0),
                    self.first_ages,
                )
            else:
                entry = np.insert(self.ages_at_events, self.first_age_index, 0.0)
        else:
            time = self.ages_at_events.copy()
            _ids = self.events_assets_ids.copy()
            if self.first_ages is not None:
                entry = np.roll(self.ages_at_events, 1)
                entry[self.first_age_index] = self.first_ages
            else:
                entry = np.roll(self.ages_at_events, 1)
                entry[self.first_age_index] = 0.0
        model_args = (
            tuple(np.take(arg, _ids) for arg in self.model_args)
            if self.model_args is not None
            else ()
        )
        return time, event, entry, model_args
