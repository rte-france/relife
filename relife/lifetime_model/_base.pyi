from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Generic,
    Literal,
    Optional,
    Self,
    overload, Union,
)

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds

from relife import FrozenParametricModel as FrozenParametricModel
from relife import ParametricModel as ParametricModel
from relife._typing import _B, _X, _Y, _Xs
from relife.likelihood import FittingResults as FittingResults

from ._plot import PlotParametricLifetimeModel

class ParametricLifetimeModel(ParametricModel, ABC, Generic[*_Xs]):
    @overload
    def sf(self, time: _X, *args: *_Xs) -> _Y: ...
    @abstractmethod
    def hf(self, time: _X, *args: *_Xs) -> _Y: ...
    @abstractmethod
    def chf(self, time: _X, *args: *_Xs) -> _Y: ...
    @abstractmethod
    def pdf(self, time: _X, *args: *_Xs) -> _Y: ...
    def cdf(self, time: _X, *args: *_Xs) -> _Y: ...
    def ppf(self, probability: _X, *args: *_Xs) -> _Y: ...
    def median(self, *args: *_Xs) -> _Y: ...
    def isf(self, probability: _X, *args: *_Xs) -> _Y: ...
    def ichf(self, cumulative_hazard_rate: _X, *args: *_Xs) -> _Y: ...
    def moment(self, n: int, *args: *_Xs) -> _Y: ...
    def mean(self, *args: *_Xs) -> _Y: ...
    def var(self, *args: *_Xs) -> _Y: ...
    def mrl(self, time: _X, *args: *_Xs) -> _Y: ...
    def ls_integrate(self, func: Callable[[_X], _Y], a: _X, b: _X, *args: *_Xs, deg: int = 10) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _Y]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
    @property
    def plot(self) -> PlotParametricLifetimeModel: ...

class FittableParametricLifetimeModel(ParametricLifetimeModel[*_Xs], ABC):
    fitting_results = Optional[FittingResults]

    def __init__(self, **kwparams: Optional[float]) -> None: ...
    @abstractmethod
    def dhf(self, time: _X, *args: *_Xs) -> _Y: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_hf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    @abstractmethod
    def jac_hf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_chf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    @abstractmethod
    def jac_chf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_sf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_sf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    @abstractmethod
    def jac_sf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @overload
    def jac_pdf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[False] = False,
    ) -> tuple[_Y, ...]: ...
    @overload
    def jac_pdf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: Literal[True] = True,
    ) -> _Y: ...
    @abstractmethod
    def jac_pdf(
        self,
        time: _X,
        *args: *_Xs,
        asarray: bool = True,
    ) -> tuple[_Y, ...] | _Y: ...
    @abstractmethod
    def _get_params_bounds(self) -> Bounds: ...
    @abstractmethod
    def _get_initial_params(
        self,
        time: NDArray[np.float64],
        *args: *_Xs,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]: ...
    def fit(
        self,
        time: NDArray[np.float64],
        *args: *_Xs,
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
        **options: Any,
    ) -> None: ...

class NonParametricLifetimeModel(ABC):
    @abstractmethod
    def fit(
        self,
        time: NDArray[np.float64],
        event: Optional[NDArray[np.bool_]] = None,
        entry: Optional[NDArray[np.float64]] = None,
        departure: Optional[NDArray[np.float64]] = None,
    ) -> Self: ...

class FrozenParametricLifetimeModel(FrozenParametricModel[*_Xs]):

    unfrozen_model: ParametricLifetimeModel[*_Xs]

    def __init__(self, model: ParametricLifetimeModel[*_Xs], *args: *_Xs) -> None: ...
    def hf(self, time: _X) -> _Y: ...
    def chf(self, time: _X) -> _Y: ...
    def sf(self, time: _X) -> _Y: ...
    def pdf(self, time: _X) -> _Y: ...
    def mrl(self, time: _X) -> _Y: ...
    def moment(self, n: int) -> _Y: ...
    def mean(self) -> _Y: ...
    def var(self) -> _Y: ...
    def isf(self, probability: _X) -> _Y: ...
    def ichf(self, cumulative_hazard_rate: _X) -> _Y: ...
    def cdf(self, time: _X) -> _Y: ...
    def ppf(self, probability: _X) -> _Y: ...
    def median(self) -> _Y: ...
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        deg: int = 10,
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B]: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _Y]: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
