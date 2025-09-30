from typing import Callable, Literal, Optional, Union, overload, TypeAlias

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife._io_types import (
    _AdditionalIntOrFloatValues,
    _BooleanValues,
    _IntOrFloatValues,
    _NumpyFloatValues,
)

from ._base import ParametricLifetimeModel
from ._type import _ParametricLifetimeModel_Proto, _FrozenParametricLifetimeModel_Proto

__all__ = ["AgeReplacementModel", "LeftTruncatedModel"]

def _reshape_ar_or_a0(name: str, value: _IntOrFloatValues) -> NDArray[np.float64]: ...

_Ar: TypeAlias = _IntOrFloatValues
_A0 : TypeAlias = _IntOrFloatValues

class AgeReplacementModel(ParametricLifetimeModel[*tuple[_Ar, *_AdditionalIntOrFloatValues]]):
    # here baseline can be freeze as _ParametricLifetimeModel_Proto[()] := _FrozenParametricLifetimeModel_Proto[*FrozenValues]
    baseline: _ParametricLifetimeModel_Proto[*_AdditionalIntOrFloatValues]
    def __init__(self, baseline: ParametricLifetimeModel[*_AdditionalIntOrFloatValues]) -> None: ...
    def sf(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def hf(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def chf(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def cdf(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def pdf(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def isf(
        self, probability: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ppf(
        self, probability: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def median(self, ar: _Ar, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mrl(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def moment(self, n: int, ar: _Ar, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mean(self, ar: _Ar, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def var(self, ar: _Ar, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Ar,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Ar,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Ar,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _NumpyFloatValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Ar,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
        ar: _Ar,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> (
        _NumpyFloatValues
        | tuple[_NumpyFloatValues, _NumpyFloatValues]
        | tuple[_NumpyFloatValues, _BooleanValues]
        | tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]
    ): ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        ar: _Ar,
        *args: *_AdditionalIntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    def freeze(
        self, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _FrozenParametricLifetimeModel_Proto[*tuple[_Ar, *_AdditionalIntOrFloatValues]]: ...

class LeftTruncatedModel(ParametricLifetimeModel[*tuple[_A0, *_AdditionalIntOrFloatValues]]):
    baseline: _ParametricLifetimeModel_Proto[*_AdditionalIntOrFloatValues]
    def __init__(self, baseline: ParametricLifetimeModel[*_AdditionalIntOrFloatValues]) -> None: ...
    def sf(
        self, time: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def hf(
        self, time: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def chf(
        self, time: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def cdf(
        self, time: _IntOrFloatValues, ar: _Ar, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def pdf(
        self, time: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def isf(
        self, probability: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def ppf(
        self, probability: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def median(self, ar: _A0, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    @override
    def mrl(
        self, time: _IntOrFloatValues, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    @override
    def moment(self, n: int, ar: _A0, *args: *_AdditionalIntOrFloatValues) -> NDArray[np.float64]: ...
    @override
    def mean(self, ar: _A0, *args: *_AdditionalIntOrFloatValues) -> NDArray[np.float64]: ...
    @override
    def var(self, ar: _A0, *args: *_AdditionalIntOrFloatValues) -> NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _A0,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _A0,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _A0,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _NumpyFloatValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _A0,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
        ar: _A0,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> (
        _NumpyFloatValues
        | tuple[_NumpyFloatValues, _NumpyFloatValues]
        | tuple[_NumpyFloatValues, _BooleanValues]
        | tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]
    ): ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        ar: _A0,
        *args: *_AdditionalIntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    def freeze(
        self, ar: _A0, *args: *_AdditionalIntOrFloatValues
    ) -> _FrozenParametricLifetimeModel_Proto[*tuple[_A0, _AdditionalIntOrFloatValues]]: ...
