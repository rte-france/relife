import copy
from typing import Generic, Optional, Callable, NewType, TypeVarTuple, Union, Type, Self

import numpy as np
from numpy.typing import NDArray

from relife.decorators import isbroadcastable
from relife.distributions import LifetimeDistribution
from relife.parametric import (
    Regression,
    LeftTruncatedDistribution,
    AgeReplacementDistribution,
    Distribution,
)

Z = TypeVarTuple("Z")
T = NewType("T", NDArray[np.floating] | NDArray[np.integer] | float | int)
Covar = NewType("Covar", NDArray[np.floating] | NDArray[np.integer] | float | int)
A0 = NewType("A0", NDArray[np.floating] | NDArray[np.integer] | float | int)
Ar = NewType("Ar", NDArray[np.floating] | NDArray[np.integer] | float | int)


def _reshape(*z: *Z):
    nb_assets = 1  # minimum value
    for arr in z:
        arr = np.asarray(arr)
        if arr.ndim > 2:
            raise ValueError("Number of dimension can't be higher than 2 in zvariables")
        if arr.size == 1:
            yield np.squeeze(arr).item()  # yield float
        else:
            arr = arr.reshape(-1, 1)
            if (
                nb_assets != 1 and arr.shape[0] != nb_assets
            ):  # test if nb assets changed
                raise ValueError("Different number of assets are given in zvariables")
            else:  # update nb_assets
                nb_assets = arr.shape[0]
            yield arr.reshape(-1, 1)


def _zvar_name(baseline: LifetimeDistribution[*Z]):
    if isinstance(baseline, Regression):
        return "covar"
    if isinstance(baseline, AgeReplacementDistribution):
        return "ar"
    if isinstance(baseline, LeftTruncatedDistribution):
        return "a0"
    if isinstance(baseline, Distribution):
        return


class UnivariateLifetimeDistribution(Generic[*Z]):

    univariate: bool = True

    def __init__(
        self,
        baseline: Union[LifetimeDistribution[*Z], Type[Self][*Z]],
        *z: *Z,
    ):

        self.zvariables = {}
        if isinstance(baseline, UnivariateLifetimeDistribution):
            z = (*z, *baseline.zvalues)
            self.zvariables = copy.deepcopy(baseline.zvariables)

        self.baseline = baseline

        z = tuple(_reshape(*z))
        self.nb_assets = max(
            map(lambda x: x.shape[0] if isinstance(x, np.ndarray) else 1, iter(z)),
            default=1,
        )
        # TODO : contruct dict[name : str, value] to zvar name and catch them more easily later
        # add __getattr__ to catch automatically those zvar value from their name

        if len(z) > 0:
            i = 0
            zvar_name = _zvar_name(baseline)
            if zvar_name is not None:
                self.zvariables[zvar_name] = z[i]
            while hasattr(baseline, "baseline"):
                baseline = baseline.baseline
                zvar_name = _zvar_name(baseline)
                if zvar_name is not None:
                    i += 1
                    self.zvariables[zvar_name] = z[i]

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("zvariables"):
            return super().__getattribute__("zvariables")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    @property
    def zvalues(self):
        return tuple(self.zvariables.values())

    @isbroadcastable("time")
    def hf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.hf(time, *self.zvalues)

    @isbroadcastable("time")
    def chf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.chf(time, *self.zvalues)

    @isbroadcastable("time")
    def sf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.sf(time, *self.zvalues)

    @isbroadcastable("time")
    def pdf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.pdf(time, *self.zvalues)

    @isbroadcastable("time")
    def mrl(self, time: T) -> NDArray[np.float64]:
        return self.baseline.mrl(time, *self.zvalues)

    def moment(self, n: int) -> NDArray[np.float64]:
        return self.baseline.moment(n)

    def mean(self) -> NDArray[np.float64]:
        return self.baseline.moment(1, *self.zvalues)

    def var(self) -> NDArray[np.float64]:
        return (
            self.baseline.moment(2, *self.zvalues)
            - self.baseline.moment(1, *self.zvalues) ** 2
        )

    @isbroadcastable("probability")
    def isf(self, probability: float | NDArray[np.float64]):
        return self.baseline.isf(probability, *self.zvalues)

    @isbroadcastable("cumulative_hazard_rate")
    def ichf(self, cumulative_hazard_rate: float | NDArray[np.float64]):
        return self.baseline.ichf(cumulative_hazard_rate, *self.zvalues)

    @isbroadcastable("time")
    def cdf(self, time: T) -> NDArray[np.float64]:
        return self.baseline.cdf(time, *self.zvalues)

    def rvs(self, size: int = 1, seed: Optional[int] = None) -> NDArray[np.float64]:
        return self.baseline.rvs(*self.zvalues, size=size, seed=seed)

    @isbroadcastable("probability")
    def ppf(self, probability: float | NDArray[np.float64]) -> NDArray[np.float64]:
        return self.baseline.ppf(probability, *self.zvalues)

    def median(self) -> NDArray[np.float64]:
        return self.baseline.median(*self.zvalues)

    def ls_integrate(
        self,
        func: Callable[[NDArray[np.float64]], NDArray[np.float64]],
        a: float | NDArray[np.float64],
        b: float | NDArray[np.float64],
        deg: int = 100,
    ) -> NDArray[np.float64]:

        return self.baseline.ls_integrate(func, a, b, deg, *self.zvalues)
