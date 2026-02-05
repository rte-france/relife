# pyright: basic

from __future__ import annotations

import warnings
from typing import Any, Generic, Self, Sequence, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife.base import FrozenParametricModel, ParametricModel
from relife.data import NHPPData
from relife.lifetime_model._base import FittableParametricLifetimeModel
from relife.likelihood import DefaultLifetimeLikelihood
from relife.likelihood._base import FittingResults
from relife.stochastic_process._sample import StochasticSampleMapping
from relife.typing import AnyFloat, NumpyFloat, ScipyMinimizeOptions

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

    def intensity(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
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

    def cumulative_intensity(self, time: AnyFloat, *args: *Ts) -> NumpyFloat:
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

        from relife.utils import get_model_nb_assets

        from ._sample import NonHomogeneousPoissonProcessIterable

        frozen_nhpp = self.freeze(*args)
        iterable = NonHomogeneousPoissonProcessIterable(
            frozen_nhpp, nb_samples, time_window=time_window, seed=seed
        )
        struct_array = np.concatenate(tuple(iterable))
        struct_array = np.sort(
            struct_array, order=("asset_id", "sample_id", "timeline")
        )
        return StochasticSampleMapping.from_struct_array(
            struct_array, get_model_nb_assets(frozen_nhpp), nb_samples
        )

    def generate_failure_data(
        self, nb_samples: int, time_window: tuple[float, float], *args, seed=None
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
        optimizer_options: ScipyMinimizeOptions | None = None,
    ) -> Self:
        """
        Estimation of the process parameters from recurrent failure data.

        Parameters
        ----------
        ages_at_events : 1D array of float
            Array of float containing the ages of each asset when the events occured
        events_assets_ids : sequence of hashable
            Sequence object containing the ids of each assets corresponding to each ages.
        first_ages : 1D array of float, optional
            Array of float containing the ages of each asset before observing events. If set, ``assets_ids`` is needed and its length
            must equal the size of ``first_ages``.
        last_ages : 1D array of float, optional
            Array of float containing the ages of each asset at the end of the observation period. If set, ``assets_ids`` is needed and its length
            must equal the size of ``last_ages``.
        lifetime_model_args : tuple of np.ndarray, optional
            Additional arguments needed by the model. If set, ``assets_ids`` is needed.
            For 1D array, the size must equal the length of ``assets_ids``. For 2D array (e.g. covar of regression),
            the length of first axis must equal the length of ``assets_ids``.
        assets_ids : sequence of hashable, optional
            Only needed if either ``first_ages``, ``last_ages`` or ``model_args`` is filled. It must be a sequence object
            containing the unique ids corresponding to each values contained in ``first_ages``, ``last_ages`` and/or ``model_args``

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

        >>> from relife.lifetime_model import ProportionalHazard
        >>> nhpp = NonHomogeneousPoissonProcess(ProportionalHazard())
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
        # noinspection PyProtectedMember
        self.lifetime_model.get_initial_params(time, *args)
        likelihood = DefaultLifetimeLikelihood(
            self.lifetime_model, time, event=event, entry=entry
        )
        if optimizer_options is None:
            optimizer_options = {}
        fitting_results = likelihood.maximum_likelihood_estimation(**optimizer_options)
        self.params = fitting_results.optimal_params
        self.fitting_results = fitting_results
        return self


class FrozenNonHomogeneousPoissonProcess(
    FrozenParametricModel[NonHomogeneousPoissonProcess[*Ts], *Ts]
):
    """
    Non-homogeneous Poisson process.
    """

    def intensity(self, time: AnyFloat) -> NumpyFloat:
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

    def cumulative_intensity(self, time: AnyFloat) -> NumpyFloat:
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
        return self._unfrozen_model.sample(
            nb_samples, time_window, *self.args, seed=seed
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
