# """Lifetime random variate sampling utilities."""
#
# from collections.abc import Sequence
# from typing import Literal, NamedTuple, TypeAlias, TypeVarTuple, overload
#
# import numpy as np
# from optype.numpy import Array, Array1D
#
# from relife.stochastic_processes import RenewalProcess
#
# ST: TypeAlias = int | float
# NumpyST: TypeAlias = np.floating | np.uint
# Ts = TypeVarTuple("Ts")
# AgeArg: TypeAlias = ST | NumpyST | Array1D[NumpyST] | None
# Seed: TypeAlias = (
#     int | np.random.Generator | np.random.BitGenerator | np.random.RandomState | None
# )
#
#
# class LifetimeFitArgs(NamedTuple):
#     time: Array1D[np.float64] | Array[tuple[int, Literal[2]], np.float64]
#     event: Array1D[np.bool_] | None = None
#     entry: Array1D[np.float64] | None = None
#     args: Array1D[np.float64] | Sequence[Array1D[np.float64]] | None = None
#
#
# @overload
# def sample_lifetimes(
#     model,
#     size: int | tuple[int, ...] | None = None,
#     a0: AgeArg = None,
#     ar: AgeArg = None,
#     seed: Seed = None,
# ) -> LifetimeFitArgs: ...
# @overload
# def sample_lifetimes(
#     model,
#     size: int | tuple[int, ...] | None = None,
#     a0: AgeArg = None,
#     ar: AgeArg = None,
#     seed: Seed = None,
# ) -> LifetimeFitArgs: ...
# def sample_lifetimes(
#     model,
#     size: int | tuple[int, ...] | None = None,
#     a0: AgeArg = None,
#     ar: AgeArg = None,
#     seed: Seed = None,
# ) -> LifetimeFitArgs:
#     pass
#
#
# def _sample_lifetimes(
#     model,
#     size: int | tuple[int, ...] | None = None,
#     *args: *Ts,
#     seed: int
#     | np.random.Generator
#     | np.random.BitGenerator
#     | np.random.RandomState
#     | None = None,
# ) -> np.float64 | ArrayND[np.float64]:  # must return time, event, entry, args
#     """
#     Sample lifetimes from a model using inverse survival transform.
#
#     Parameters
#     ----------
#     model : InverseSurvivalModel
#         Model exposing an ``isf`` method.
#     size : int or tuple of int, optional
#         Size of the generated sample.
#     *args
#         Additional arguments passed to ``model.isf``.
#     seed : int, np.random.BitGenerator, np.random.Generator, np.random.RandomState, optional
#         Random number generator seed or instance.
#
#     Returns
#     -------
#     float or ndarray
#         Sampled lifetimes.
#     """  # noqa: E501
#     rng = np.random.default_rng(seed)
#     probability = rng.uniform(0.0, 1.0, size)
#     return model.isf(probability, *args)
#
#
# def generate_renewal_process_lifetime_data(
#     process: RenewalProcess,
#     nb_samples: int,
#     time_window: tuple[float, float],
#     a0: AgeArg = None,
#     ar: AgeArg = None,
#     seed: Seed = None,
# ) -> LifetimeFitArgs:
#     """Generate lifetime data from a renewal process sample."""
#     if process.first_lifetime_model:
#         raise ValueError(
#             "Calling sample_lifetime_data with first_lifetime_model is ambiguous."
#         )
#
#     iterable = RenewalProcessIterable(
#         process, nb_samples, time_window, a0=a0, ar=ar, seed=seed
#     )
#     struct_array = np.concatenate(tuple(iterable))
#     struct_array = np.sort(struct_array, order=("sample_id", "asset_id", "timeline"))
#
#     args_2d = tuple(
#         np.atleast_2d(arg) for arg in getattr(process.lifetime_model, "args", ())
#     )
#     tuple_args_arr = tuple(
#         np.take(np.asarray(arg), struct_array["asset_id"], axis=0) for arg in args_2d
#     )
#
#     return LifetimeFitArgs(
#         time=struct_array["time"].copy(),
#         event=struct_array["event"].copy(),
#         entry=struct_array["entry"].copy(),
#         args=tuple_args_arr,
#     )
