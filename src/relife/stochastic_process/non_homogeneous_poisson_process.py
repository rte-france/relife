from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Self, TypeVarTuple

import numpy as np
from numpy.typing import NDArray
from optype.numpy import Array, Array1D, AtMost2D

from relife.base import FittingResults, FrozenParametricModel, ParametricModel
from relife.lifetime_model._base import (
    FittableParametricLifetimeModel,
)
from relife.stochastic_process._sample import StochasticSampleMapping

Ts = TypeVarTuple("Ts")

__all__ = ["NonHomogeneousPoissonProcess", "FrozenNonHomogeneousPoissonProcess"]


class NonHomogeneousPoissonProcess(ParametricModel, Generic[*Ts]):
    """
    Non-homogeneous Poisson process.
    """

    lifetime_model: FittableParametricLifetimeModel[*Ts]
    fitting_results: FittingResults | None

    def __init__(self, lifetime_model: FittableParametricLifetimeModel[*Ts]):
        super().__init__()
        self.lifetime_model = lifetime_model
        self.fitting_results = None

    def intensity(
        self, time: int | float | Array[AtMost2D, np.float64], *args: *Ts
    ) -> np.float64 | Array[AtMost2D, np.float64]:
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
        self, time: int | float | Array[AtMost2D, np.float64], *args: *Ts
    ) -> np.float64 | Array[AtMost2D, np.float64]:
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

    def freeze(self, *args: *Ts) -> FrozenNonHomogeneousPoissonProcess[*Ts]:
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
        *args: *Ts,
        a0: int | float | Array1D[np.float64] | None = None,
        ar: int | float | Array1D[np.float64] | None = None,
        seed=None,
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
        self, nb_samples: int, time_window: tuple[float, float], *args: *Ts, seed=None
    ) -> dict[str, Any]:
        r"""
        .. warning:: Not implemented yet
        """
        raise NotImplementedError(
            "Failure data methods for stochastic processes will be introduced in a future release"
        )

        # from ._sample import NonHomogeneousPoissonProcessIterable

        # frozen_nhpp = self.freeze(*args)

        # iterable = NonHomogeneousPoissonProcessIterable(
        #     frozen_nhpp, nb_samples, time_window=time_window, seed=seed
        # )
        # struct_array = np.concatenate(tuple(iterable))
        # struct_array = np.sort(
        #     struct_array, order=("sample_id", "asset_id", "timeline")
        # )

        # first_ages_index = np.nonzero(struct_array["entry"] == time_window[0])
        # last_ages_index = np.nonzero(struct_array["age"] == time_window[1])

        # event_index = np.nonzero(struct_array["event"])

        # first_ages = struct_array[first_ages_index]["entry"].copy()
        # last_ages = struct_array[last_ages_index]["age"].copy()

        # assets_ids = np.char.add(
        #     np.char.add(
        #         np.full_like(
        #             struct_array[last_ages_index]["sample_id"], "S", dtype=np.str_
        #         ),
        #         struct_array[last_ages_index]["sample_id"].astype(np.str_),
        #     ),
        #     np.char.add(
        #         np.full_like(
        #             struct_array[last_ages_index]["asset_id"], "A", dtype=np.str_
        #         ),
        #         struct_array[last_ages_index]["asset_id"].astype(np.str_),
        #     ),
        # )

        # events_assets_ids = np.char.add(
        #     np.char.add(
        #         np.full_like(
        #             struct_array[event_index]["sample_id"], "S", dtype=np.str_
        #         ),
        #         struct_array[event_index]["sample_id"].astype(np.str_),
        #     ),
        #     np.char.add(
        #         np.full_like(struct_array[event_index]["asset_id"], "A", dtype=np.str_),
        #         struct_array[event_index]["asset_id"].astype(np.str_),
        #     ),
        # )
        # ages_at_events = struct_array[event_index]["age"].copy()

        # return {
        #     "ages_at_events": ages_at_events,
        #     "events_assets_ids": events_assets_ids,
        #     "first_ages": first_ages,
        #     "last_ages": last_ages,
        #     "assets_ids": assets_ids,
        # }

    def fit(
        self,
        ages_at_events: NDArray[np.float64],
        events_assets_ids: Sequence[str] | NDArray[np.int64],
        first_ages: NDArray[np.float64] | None = None,
        last_ages: NDArray[np.float64] | None = None,
        lifetime_model_args: NDArray[Any] | tuple[NDArray[Any], ...] | None = None,
        assets_ids: Sequence[str] | NDArray[np.int64] | None = None,
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
        """
        warnings.warn(
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
        optimizer = self.lifetime_model.init_likelihood(
            time, args, event, entry, **kwargs
        )
        fitting_results = optimizer.optimize()
        self.set_params(fitting_results.optimal_params)
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(
    FrozenParametricModel[NonHomogeneousPoissonProcess[*Ts], *Ts]
):
    """
    Non-homogeneous Poisson process.
    """

    def intensity(
        self, time: int | float | Array[AtMost2D, np.float64]
    ) -> np.float64 | Array[AtMost2D, np.float64]:
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
        return self._unfrozen_model.intensity(time, *self.args)

    def cumulative_intensity(
        self, time: int | float | Array[AtMost2D, np.float64]
    ) -> np.float64 | Array[AtMost2D, np.float64]:
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
        return self._unfrozen_model.cumulative_intensity(time, *self.args)

    def sample(
        self,
        nb_samples: int,
        time_window: tuple[float, float],
        a0: int | float | Array1D[np.float64] | None = None,
        ar: int | float | Array1D[np.float64] | None = None,
        seed=None,
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
        return self._unfrozen_model.sample(
            nb_samples, time_window, *self.args, a0=a0, ar=ar, seed=seed
        )

    def generate_failure_data(
        self, nb_samples: int, time_window: tuple[float, float], seed=None
    ):
        """Generate failure data

        This function will generate failure data that can be used to fit a non-homogeneous Poisson process.

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
        """
        return self._unfrozen_model.generate_failure_data(
            nb_samples, time_window, *self.args, seed=seed
        )


def is_non_homogeneous_poisson_process(model):
    """
    Checks if model is a non-homogeneous Poisson process.
    """
    # local import to avoid circular import

    return isinstance(
        model, (NonHomogeneousPoissonProcess, FrozenNonHomogeneousPoissonProcess)
    )


@dataclass
class NHPPData:
    ages_at_events: NDArray[np.float64]
    events_assets_ids: Sequence[str] | NDArray[np.int64]
    first_ages: NDArray[np.float64] | None = field(repr=False, default=None)
    last_ages: NDArray[np.float64] | None = field(repr=False, default=None)
    model_args: tuple[float | NDArray[np.float64], ...] | None = field(
        repr=False, default=None
    )
    assets_ids: Sequence[str] | NDArray[np.int64] | None = field(
        repr=False, default=None
    )

    first_age_index: NDArray[np.int64] = field(repr=False, init=False)
    last_age_index: NDArray[np.int64] = field(repr=False, init=False)

    def __post_init__(self):
        # convert inputs to arrays
        self.events_assets_ids = np.unique(
            np.asarray(self.events_assets_ids), return_inverse=True
        )[1].astype(np.uint32)
        self.ages_at_events = np.asarray(self.ages_at_events, dtype=np.float64)
        if self.assets_ids is not None:
            self.assets_ids = np.unique(
                np.asarray(self.assets_ids), return_inverse=True
            )[1].astype(np.uint32)
        if self.first_ages is not None:
            self.first_ages = np.asarray(self.first_ages, dtype=np.float64)
        if self.last_ages is not None:
            self.last_ages = np.asarray(self.last_ages, dtype=np.float64)

        # control shapes
        if self.events_assets_ids.ndim != 1:
            raise ValueError(
                "Invalid array shape for events_assets_ids. Expected 1d-array"
            )
        if self.ages_at_events.ndim != 1:
            raise ValueError("Invalid array shape for ages. Expected 1d-array")
        if len(self.events_assets_ids) != len(self.ages_at_events):
            raise ValueError(
                "Shape of events_assets_ids and ages must be equal. Expected equal length 1d-arrays"
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
                        "Shape of assets_ids and start_ages must be equal. Expected equal length 1d-arrays"
                    )
            if self.last_ages is not None:
                if self.last_ages.ndim != 1:
                    raise ValueError(
                        "Invalid array shape for last_ages. Expected 1d-array"
                    )
                if len(self.last_ages) != len(self.assets_ids):
                    raise ValueError(
                        "Shape of assets_ids and last_ages must be equal. Expected equal length 1d-arrays"
                    )
            if bool(self.model_args):
                for arg in self.model_args:
                    arg = np.atleast_2d(np.asarray(arg, dtype=np.float64))
                    if arg.ndim > 2:
                        raise ValueError(
                            "Invalid arg shape in model_args. Arrays must be 0, 1 or 2d"
                        )
                    try:
                        arg.reshape((len(self.assets_ids), -1))
                    except ValueError:
                        raise ValueError(
                            """
                            Invalid arg shape in model_args. Arrays must
                            coherent with the number of assets given by
                            assets_ids
                            """
                        )
        else:
            if self.first_ages is not None:
                raise ValueError(
                    "If first_ages is given, corresponding asset ids must be given in assets_ids"
                )
            if self.last_ages is not None:
                raise ValueError(
                    "If last_ages is given, corresponding asset ids must be given in assets_ids"
                )
            if bool(self.model_args):
                raise ValueError(
                    "If model_args is given, corresponding asset ids must be given in assets_ids"
                )

        # if self.events_assets_ids.dtype != np.int64:
        #     events_assets_ids = np.unique(self.events_assets_ids, return_inverse=True)[1]
        # # convert assets_id to int id
        # if self.assets_ids is not None:
        #     if self.assets_ids.dtype != np.int64:
        #         assets_ids = np.unique(self.assets_ids, return_inverse=True)[1]
        #     # control ids correspondance
        #     if not np.all(np.isin(self.events_assets_ids, self.assets_ids)):
        #         raise ValueError("If assets_ids is filled, all values of events_assets_ids must exist in assets_ids")

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

            if self.first_ages is not None:
                if np.any(
                    self.ages_at_events[self.first_age_index]
                    <= self.first_ages[nb_ages_per_asset != 0]
                ):
                    raise ValueError(
                        "Each first_ages value must be lower than all of its corresponding ages values"
                    )
            if self.last_ages is not None:
                if np.any(
                    self.ages_at_events[self.last_age_index]
                    >= self.last_ages[nb_ages_per_asset != 0]
                ):
                    raise ValueError(
                        "Each last_ages value must be greater than all of its corresponding ages values"
                    )

    def to_lifetime_data(self):
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
