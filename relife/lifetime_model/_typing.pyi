from typing import Callable, Literal, Optional, Protocol, Union, overload

import numpy as np

from relife._typing import (
    _AdditionalIntOrFloatValues,
    _BooleanValues,
    _IntOrFloatValues,
    _NumpyFloatValues,
)

class _ParametricLifetimeModel(Protocol[*_AdditionalIntOrFloatValues]):
    def hf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def sf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def pdf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def mrl(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def moment(self, n: int, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def mean(self, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def var(self, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def isf(self, probability: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def ichf(
        self, cumulative_hazard_rate: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues
    ) -> _NumpyFloatValues: ...
    def cdf(self, time: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def ppf(self, probability: _IntOrFloatValues, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def median(self, *args: *_AdditionalIntOrFloatValues) -> _NumpyFloatValues: ...
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        *args: *_AdditionalIntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _NumpyFloatValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
        *args: *_AdditionalIntOrFloatValues,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        random_state: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> (
        _NumpyFloatValues
        | tuple[_NumpyFloatValues, _NumpyFloatValues]
        | tuple[_NumpyFloatValues, _BooleanValues]
        | tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]
    ): ...

class _FrozenParametricLifetimeModel(Protocol[*_AdditionalIntOrFloatValues]):

    unfrozen_model: _ParametricLifetimeModel[*_AdditionalIntOrFloatValues]

    def hf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def chf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def sf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def pdf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def mrl(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def moment(self, n: int) -> _NumpyFloatValues: ...
    def mean(self) -> _NumpyFloatValues: ...
    def var(self) -> _NumpyFloatValues: ...
    def isf(self, probability: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def ichf(self, cumulative_hazard_rate: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def cdf(self, time: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def ppf(self, probability: _IntOrFloatValues) -> _NumpyFloatValues: ...
    def median(self) -> _NumpyFloatValues: ...
    def ls_integrate(
        self,
        func: Callable[[_IntOrFloatValues], _NumpyFloatValues],
        a: _IntOrFloatValues,
        b: _IntOrFloatValues,
        deg: int = 10,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> _NumpyFloatValues: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _NumpyFloatValues]: ...
    @overload
    def rvs(
        self,
        size: int,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_NumpyFloatValues, _BooleanValues, _NumpyFloatValues]: ...
    def rvs(
        self,
        size: int,
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
