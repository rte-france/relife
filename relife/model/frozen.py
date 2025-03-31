import functools
from typing import Optional, Callable, NewType, Union, Self

import numpy as np
from numpy.typing import NDArray

from relife.model import LifetimeModel
from relife.parametric_model import (
    Regression,
    LeftTruncatedModel,
    AgeReplacementModel,
    Distribution,
)
from relife.sampling import CountData
from relife.stochastic_process import NonHomogeneousPoissonProcess

T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)
ModelArgs = NewType(
    "ModelArgs", NDArray[np.floating] | NDArray[np.integer] | float | int
)


def _make_model_args(model_args: dict[str, ModelArgs]):
    nb_assets = 1  # minimum value
    for k, v in model_args.items():
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
            model_args[k] = arr
        else:
            arr = arr.reshape(-1, 1)
            if (
                nb_assets != 1 and arr.shape[0] != nb_assets
            ):  # test if nb assets changed
                raise ValueError("Different number of assets are given in model args")
            else:  # update nb_assets
                nb_assets = arr.shape[0]
            model_args[k] = arr
    return nb_assets, model_args


def _get_args_names(
    baseline: LifetimeModel[*tuple[ModelArgs, ...]],
) -> tuple[str, ...]:

    def arg_name(obj: LifetimeModel[*tuple[ModelArgs, ...]]) -> tuple[str, ...]:
        if isinstance(obj, Regression):
            return ("covar",)
        if isinstance(obj, AgeReplacementModel):
            return ("ar",)
        if isinstance(obj, LeftTruncatedModel):
            return ("a0",)
        if isinstance(obj, Distribution):
            return ()

    args_names = []
    args_names.extend(arg_name(baseline))
    while hasattr(baseline, "baseline"):
        baseline = baseline.baseline
        args_names.extend(arg_name(baseline))

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


class FrozenLifetimeModel:

    frozen: bool = True

    def __init__(
        self,
        baseline: Union[LifetimeModel[*tuple[ModelArgs, ...]], Self],
        **kwargs: ModelArgs,
    ):
        self.model_args = {}
        model_args = {}
        model_args.update(kwargs)
        if isinstance(baseline, FrozenLifetimeModel):
            model_args.update(baseline.model_args)
        args_names = _get_args_names(baseline)
        if tuple(model_args.keys()) != args_names:
            raise ValueError(f"Expected {args_names} kw arguments")
        nb_assets, model_args = _make_model_args(model_args)

        self.baseline = baseline
        self.model_args = model_args
        self.nb_assets = nb_assets

    @property
    def args(self):
        return tuple(self.model_args.values())

    @isbroadcastable("time")
    def hf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.hf(time, *self.args)

    @isbroadcastable("time")
    def chf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.chf(time, *self.args)

    @isbroadcastable("time")
    def sf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.sf(time, *self.args)

    @isbroadcastable("time")
    def pdf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.pdf(time, *self.args)

    @isbroadcastable("time")
    def mrl(self, time: T) -> NDArray[np.float64]:
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
    def cdf(self, time: T) -> NDArray[np.float64]:
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
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.baseline.ls_integrate(func, a, b, deg, *self.args)


class FrozenNonHomogeneousPoissonProcess:

    def __init__(
        self,
        baseline: NonHomogeneousPoissonProcess[tuple[ModelArgs, ...]],
        **kwargs: ModelArgs,
    ):
        self.model_args = {}
        model_args = {}
        model_args.update(kwargs)
        if isinstance(baseline, FrozenLifetimeModel):
            model_args.update(baseline.model_args)
        args_names = _get_args_names(baseline)
        if tuple(model_args.keys()) != args_names:
            raise ValueError(f"Expected {args_names} kw arguments")
        nb_assets, model_args = _make_model_args(model_args)

        self.baseline = baseline
        self.model_args = model_args
        self.nb_assets = nb_assets

    @property
    def args(self):
        return tuple(self.model_args.values())

    def intensity(self, time: T) -> NDArray[np.float64]:
        return self.baseline.intensity(time, *self.args)

    def cumulative_intensity(self, time: T) -> NDArray[np.float64]:
        return self.baseline.cumulative_intensity(time, *self.args)

    def sample(
        self,
        size: int,
        tf: float,
        t0: float = 0.0,
        maxsample: int = 1e5,
        seed: Optional[int] = None,
    ) -> CountData:
        from relife.sampling import sample_count_data

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
        from relife.sampling import failure_data_sample

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
