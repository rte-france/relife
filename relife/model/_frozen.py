from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Callable, Optional, TypeVarTuple, Generator

import numpy as np
from numpy.typing import NDArray

from relife._plots import PlotConstructor, PlotNHPP

if TYPE_CHECKING:
    from relife.sample import CountData
    from relife.stochastic_process import NonHomogeneousPoissonProcess

    from ._parametric import ParametricModel
    from ._protocol import LifetimeModel

Args = TypeVarTuple("Args")

from relife.parametric_model import (
    AFT,
    AgeReplacementModel,
    LeftTruncatedModel,
    ProportionalHazard,
)


def _args_names(
    model: ParametricModel,
) -> Generator[str]:
    match model:
        case (ProportionalHazard(), AFT()):
            yield "covar"
            return _args_names(model.baseline)
        case AgeReplacementModel():
            yield "ar"
            return _args_names(model.baseline)
        case LeftTruncatedModel():
            yield "a0"
            return _args_names(model.baseline)
        case _:  # in other case, stop generator and yield nothing
            return


def _reshape(arg_name: str, arg_value: float | NDArray[np.float64]) -> float | NDArray[np.float64]:
    arg_value = np.asarray(arg_value)
    ndim = arg_value.ndim
    if ndim > 2:
        raise ValueError(
            f"Number of dimension can't be higher than 2. Got {ndim} for {arg_name}"
        )
    match arg_name:
        case "covar":
            if arg_value.ndim <= 1:
                return arg_value.reshape(1, -1)
            return arg_value
        case ("a0", "ar"):
            if arg_value.ndim <= 1:
                if arg_value.size == 1:
                    return arg_value.item()
                return arg_value.reshape(-1, 1)
            return arg_value


def _make_kwargs(
    model: ParametricModel, *args: float | NDArray[np.float64]
) -> dict[str, float | NDArray[np.float64]]:

    args_names = tuple(_args_names(model))
    if len(args_names) != len(args):
        raise TypeError(
            f"{model.__class__.__name__}.freeze() requires {args_names} positional argument but got {len(args)} argument.s only"
        )

    nb_assets = 1  # minimum value
    kwargs = {}
    for k, v in zip(args_names, args):
        v = _reshape(k, v)
        if isinstance(v, np.ndarray):  # if float, nb_assets is unchanged
            # test if nb assets changed
            if nb_assets != 1 and v.shape[0] != nb_assets:
                raise ValueError(
                    "Different number of assets are passed through arguments"
                )
            # update nb_assets
            else:
                nb_assets = v.shape[0]
        kwargs[k] = v
    return kwargs


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


class FrozenParametricModel:

    frozen: bool = True

    def __init__(
        self,
        model: ParametricModel,
        *args: float | NDArray[np.float64],
    ):
        self.model = model
        self.kwargs = _make_kwargs(model, *args)

    @property
    def args(self) -> tuple[float | NDArray[np.float64], ...]:
        return tuple(self.kwargs.values())

    @property
    def nb_assets(self) -> int:
        return max(
            map(
                lambda x: np.asarray(x).shape[0] if x.ndim >= 1 else 1,
                self.kwargs.values(),
            ),
            default=1,
        )


# better with FrozenLifetimeModel and freeze in LifetimeModel (match with AgeReplacementModel and LeftTruncatedModel)
class FrozenLifetimeModel(FrozenParametricModel):
    model: LifetimeModel[*tuple[float | NDArray, ...]]

    @isbroadcastable("time")
    def hf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.hf(time, *self.args)

    @isbroadcastable("time")
    def chf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.chf(time, *self.args)

    @isbroadcastable("time")
    def sf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.sf(time, *self.args)

    @isbroadcastable("time")
    def pdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.pdf(time, *self.args)

    @isbroadcastable("time")
    def mrl(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.mrl(time, *self.args)

    def moment(self, n: int) -> NDArray[np.float64]:
        return self.model.moment(n)

    def mean(self) -> NDArray[np.float64]:
        return self.model.moment(1, *self.args)

    def var(self) -> NDArray[np.float64]:
        return self.model.moment(2, *self.args) - self.model.moment(1, *self.args) ** 2

    @isbroadcastable("probability")
    def isf(self, probability: float | NDArray[np.float64]):
        return self.model.isf(probability, *self.args)

    @isbroadcastable("cumulative_hazard_rate")
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]):
        return self.model.ichf(cumulative_hazard_rate, *self.args)

    @isbroadcastable("time")
    def cdf(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.cdf(time, *self.args)

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> NDArray[np.float64]:
        return self.model.rvs(*self.args, size=size, seed=seed)

    @isbroadcastable("probability")
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.ppf(probability, *self.args)

    def median(self) -> NDArray[np.float64]:
        return self.model.median(*self.args)

    def ls_integrate(
        self,
        func: Callable[[float | NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.model.ls_integrate(func, a, b, deg, *self.args)


class FrozenNonHomogeneousPoissonProcess(FrozenParametricModel):
    model: NonHomogeneousPoissonProcess[*tuple[float | NDArray, ...]]

    def intensity(self, time: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.model.intensity(time, *self.args)

    def cumulative_intensity(
        self, time: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return self.model.cumulative_intensity(time, *self.args)

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
            self.model,
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
            self.model,
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
