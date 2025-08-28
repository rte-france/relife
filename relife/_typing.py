from typing import (
    Callable,
    Literal,
    Optional,
    Protocol,
    TypeAlias,
    TypeVarTuple,
    overload, Union,
)

import numpy as np
from numpy.typing import NDArray

_Xs = TypeVarTuple("_Xs")
_X: TypeAlias = float | NDArray[np.float64]
_Y: TypeAlias = np.float64 | NDArray[np.float64]
_B: TypeAlias = np.bool_ | NDArray[np.bool_]

# M = TypeVar("M", bound=int)
# N = TypeVar("N", bound=int)
# _Shape = TypeVarTuple("_Shape")
# _Array : TypeAlias = np.ndarray[tuple[*_Shape], np.dtype[np.float64]]
#
# _Scalar : TypeVarTuple = np.float64
# _1DArray : TypeAlias = _Array[M]
# _2DArray : TypeAlias = _Array[M, N]


class _ParametricLifetimeModel(Protocol[*_Xs]):

    def hf(self, time: _X, *args: *_Xs) -> _Y: ...
    def chf(self, time: _X, *args: *_Xs) -> _Y: ...
    def sf(self, time: _X, *args: *_Xs) -> _Y: ...
    def pdf(self, time: _X, *args: *_Xs) -> _Y: ...
    def mrl(self, time: _X, *args: *_Xs) -> _Y: ...
    def moment(self, n: int, *args: *_Xs) -> _Y: ...
    def mean(self, *args: *_Xs) -> _Y: ...
    def var(self, *args: *_Xs) -> _Y: ...
    def isf(self, probability: _X, *args: *_Xs) -> _Y: ...
    def ichf(self, cumulative_hazard_rate: _X, *args: *_Xs) -> _Y: ...
    def cdf(self, time: _X, *args: *_Xs) -> _Y: ...
    def ppf(self, probability: _X, *args: *_Xs) -> _Y: ...
    def median(self, *args: *_Xs) -> _Y: ...
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        *args: *_Xs,
        deg: int = 10,
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_Y, _B]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_Y, _Y]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
