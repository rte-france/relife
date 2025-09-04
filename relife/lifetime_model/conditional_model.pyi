from typing import Callable, Literal, Optional, overload, Union

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife._typing import _B, _X, _Y, _ParametricLifetimeModel, _Xs

from ._base import FrozenParametricLifetimeModel, ParametricLifetimeModel

def reshape_ar_or_a0(name: str, value: _X) -> NDArray[np.float64]: ...

class AgeReplacementModel(ParametricLifetimeModel[*tuple[_X, *_Xs]]):
    baseline: _ParametricLifetimeModel[*_Xs]
    def __init__(self, baseline: _ParametricLifetimeModel[*_Xs]) -> None: ...
    def sf(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    def hf(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    def chf(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    def cdf(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    def pdf(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def isf(self, probability: _X, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def ppf(self, probability: _X, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def median(self, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def mrl(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def moment(self, n: int, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def mean(self, ar: _X, *args: *_Xs) -> _Y: ...
    @override
    def var(self, ar: _X, *args: *_Xs) -> _Y: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _X,
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
        ar: _X,
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
        ar: _X,
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
        ar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        ar: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        ar: _X,
        *args: *_Xs,
        deg: int = 10,
    ) -> _Y: ...
    def freeze(self, ar: _X, *args: *_Xs) -> FrozenAgeReplacementModel[*_Xs]: ...

class LeftTruncatedModel(ParametricLifetimeModel[*tuple[_X, *_Xs]]):
    baseline: _ParametricLifetimeModel[*_Xs]
    def __init__(self, baseline: _ParametricLifetimeModel[*_Xs]) -> None: ...
    def sf(self, time: _X, a0: _X, *args: *_Xs) -> _Y: ...
    def hf(self, time: _X, a0: _X, *args: *_Xs) -> _Y: ...
    def chf(self, time: _X, a0: _X, *args: *_Xs) -> _Y: ...
    def cdf(self, time: _X, ar: _X, *args: *_Xs) -> _Y: ...
    def pdf(self, time: _X, a0: _X, *args: *_Xs) -> _Y: ...
    @override
    def isf(self, probability: _X, a0: _X, *args: *_Xs) -> _Y: ...
    @override
    def ichf(self, cumulative_hazard_rate: _X, a0: _X, *args: *_Xs) -> _Y: ...
    @override
    def ppf(self, probability: _X, a0: _X, *args: *_Xs) -> _Y: ...
    @override
    def median(self, a0: _X, *args: *_Xs) -> _Y: ...
    @override
    def mrl(self, time: _X, a0: _X, *args: *_Xs) -> _Y: ...
    @override
    def moment(self, n: int, a0: _X, *args: *_Xs) -> NDArray[np.float64]: ...
    @override
    def mean(self, a0: _X, *args: *_Xs) -> NDArray[np.float64]: ...
    @override
    def var(self, a0: _X, *args: *_Xs) -> NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        a0: _X,
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
        a0: _X,
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
        a0: _X,
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
        a0: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> tuple[_Y, _B, _Y]: ...
    def rvs(
        self,
        size: int,
        a0: _X,
        *args: *_Xs,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None
    ) -> _Y | tuple[_Y, _Y] | tuple[_Y, _B] | tuple[_Y, _B, _Y]: ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_X], _Y],
        a: _X,
        b: _X,
        a0: _X,
        *args: *_Xs,
        deg: int = 10,
    ) -> _Y: ...
    def freeze(self, a0: _X, *args: *_Xs) -> FrozenLeftTruncatedModel[*_Xs]: ...

class FrozenAgeReplacementModel(FrozenParametricLifetimeModel[*tuple[_X, *_Xs]]):
    unfrozen_model: AgeReplacementModel[*_Xs]
    args: tuple[_X, *_Xs]
    @override
    def __init__(self, model: AgeReplacementModel[*_Xs], ar: _X, *args: *_Xs) -> None: ...
    @override
    def unfreeze(self) -> AgeReplacementModel[*_Xs]: ...
    @property
    def ar(self) -> _X: ...
    # noinspection PyUnresolvedReferences
    @ar.setter
    def ar(self, value: _X) -> None: ...

class FrozenLeftTruncatedModel(FrozenParametricLifetimeModel[*tuple[_X, *_Xs]]):
    unfrozen_model: LeftTruncatedModel[*_Xs]
    args: tuple[_X, *_Xs]
    @override
    def __init__(self, model: LeftTruncatedModel[*_Xs], a0: _X, *args: *_Xs) -> None: ...
    @override
    def unfreeze(self) -> LeftTruncatedModel[*_Xs]: ...
    @property
    def a0(self) -> _X: ...
    # noinspection PyUnresolvedReferences
    @a0.setter
    def a0(self, value: _X) -> None: ...
