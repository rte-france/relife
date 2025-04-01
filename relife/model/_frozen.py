import functools
from typing import Callable, Generic, Optional, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

from relife._plots import PlotConstructor, PlotNHPP
from relife.parametric_model import (
    AgeReplacementModel,
    Distribution,
    LeftTruncatedModel,
    Regression,
)
from relife.sample import CountData
from relife.stochastic_process import NonHomogeneousPoissonProcess

from ._protocol import LifetimeModel

Args = TypeVarTuple("Args")


def _reshape_args(
    args_names: tuple[str, ...], args: tuple[float | NDArray[np.float64], ...]
) -> tuple[float | NDArray[np.float64], ...]:
    nb_assets = 1  # minimum value
    values = []
    for k, v in zip(args_names, args):
        arr = np.asarray(v)
        ndim = arr.ndim
        if ndim > 2:
            raise ValueError(
                f"Number of dimension can't be higher than 2. Got {ndim} for {k}"
            )
        min_ndim = _get_minimum_arg_ndim(k)
        if ndim < min_ndim:
            raise ValueError(
                f"Expected at least {min_ndim} dimensions for {k} but got {ndim}"
            )
        if arr.size == 1:
            values.append(arr.item())
        else:
            arr = arr.reshape(-1, 1)
            if (
                nb_assets != 1 and arr.shape[0] != nb_assets
            ):  # test if nb assets changed
                raise ValueError("Different number of assets are given in model args")
            else:  # update nb_assets
                nb_assets = arr.shape[0]
            values.append(arr)
    return tuple(values)


def _get_args_names(
    model: LifetimeModel[*Args],
) -> tuple[str, ...]:

    def arg_name(
        obj: LifetimeModel[*Args],
    ) -> tuple[str, ...]:
        if isinstance(obj, Regression):
            return ("covar",)
        if isinstance(obj, AgeReplacementModel):
            return ("ar",)
        if isinstance(obj, LeftTruncatedModel):
            return ("a0",)
        if isinstance(obj, Distribution):
            return ()

    args_names = []
    args_names.extend(arg_name(model))
    while hasattr(model, "baseline") and not model.frozen:
        model = model.baseline
        args_names.extend(arg_name(model))

    return tuple(args_names)


def _get_minimum_arg_ndim(arg_name: str):
    if arg_name == "covar":
        return 2
    return 0


def isbroadcastable(argname: str):
    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, x):
            if x.ndim == 2:
                if x.shape[0] != 1 and x.shape[0] != self.nb_assets:
                    raise ValueError(
                        f"Inconsistent {argname} shape. Got {self.nb_assets} nb of assets but got {x.shape} {argname} shape"
                    )
            return method(self, x)

        return wrapper

    return decorator


class FrozenLifetimeModel(Generic[*Args]):

    frozen: bool = True

    def __init__(
        self,
        baseline: LifetimeModel[*Args],
        *args: *Args,
    ):
        args_names = _get_args_names(baseline)
        if len(args_names) != len(args):
            raise ValueError(
                f"Expected {args_names} args but got {len(args)} args only"
            )
        args = _reshape_args(args_names, args)

        self.baseline = baseline
        self.kwargs = {k: v for (k, v) in zip(args_names, args)}

    @property
    def args(self) -> tuple[float | NDArray[np.float64], ...]:
        return self.kwargs.values()

    @property
    def nb_assets(self) -> int:
        return max(
            map(lambda x: x.shape[0] if x.ndim >= 1 else 1, self.args), default=1
        )

    @isbroadcastable("time")
    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.hf(time, *self.args)

    @isbroadcastable("time")
    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.chf(time, *self.args)

    @isbroadcastable("time")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.sf(time, *self.args)

    @isbroadcastable("time")
    def pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.pdf(time, *self.args)

    @isbroadcastable("time")
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.mrl(time, *self.args)

    def moment(self, n: int) -> NDArray[np.float64]:
        return self.baseline.moment(n)

    def mean(self) -> NDArray[np.float64]:
        return self.baseline.moment(1, *self.args)

    def var(self) -> NDArray[np.float64]:
        return (
            self.baseline.moment(2, *self.args)
            - self.baseline.moment(1, *self.args) ** 2
        )

    @isbroadcastable("probability")
    def isf(self, probability: float | NDArray[np.float64]):
        return self.baseline.isf(probability, *self.args)

    @isbroadcastable("cumulative_hazard_rate")
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]):
        return self.baseline.ichf(cumulative_hazard_rate, *self.args)

    @isbroadcastable("time")
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.cdf(time, *self.args)

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> NDArray[np.float64]:
        return self.baseline.rvs(*self.args, size=size, seed=seed)

    @isbroadcastable("probability")
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.ppf(probability, *self.args)

    def median(self) -> NDArray[np.float64]:
        return self.baseline.median(*self.args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.baseline.ls_integrate(func, a, b, deg, *self.args)


class FrozenNonHomogeneousPoissonProcess(Generic[*Args]):

    frozen: bool = True

    def __init__(
        self,
        baseline: NonHomogeneousPoissonProcess[*Args],
        *args: *Args,
    ):
        args_names = _get_args_names(baseline)
        if len(args_names) != len(args):
            raise ValueError(
                f"Expected {args_names} args but got {len(args)} args only"
            )
        args = _reshape_args(args_names, args)

        self.baseline = baseline
        self.kwargs = {k: v for (k, v) in zip(args_names, args)}

    @property
    def args(self) -> tuple[float | NDArray[np.float64], ...]:
        return self.kwargs.values()

    @property
    def nb_assets(self) -> int:
        return max(
            map(lambda x: x.shape[0] if x.ndim >= 1 else 1, self.args), default=1
        )

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.intensity(time, *self.args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.baseline.cumulative_intensity(time, *self.args)

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sample import sample_count_data

        return sample_count_data(
            self.baseline,
            size,
            tf,
            t0=t0,
            maxsample=maxsample,
            seed=seed,
        )

    def failure_data_sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> tuple[NDArray[np.float64], ...]:
        from relife.sample import failure_data_sample

        return failure_data_sample(
            self.baseline,
            size,
            tf,
            t0,
            maxsample=maxsample,
            seed=seed,
            use="model",
        )

    @property
    def plot(self) -> PlotConstructor:
        return PlotNHPP(self)
