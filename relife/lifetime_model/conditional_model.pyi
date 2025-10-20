from typing import Callable, Literal, Optional, TypeAlias, Union, overload

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

from relife._typing import _Any_Number, _Any_Numpy_Bool, _Any_Numpy_Number

from ._base import ParametricLifetimeModel

__all__ = ["AgeReplacementModel", "LeftTruncatedModel"]

from relife.base import FrozenParametricModel

def _reshape_ar_or_a0(name: str, value: _Any_Number) -> _Any_Numpy_Number: ...

_Base_Parametric_Lifetime_Model: TypeAlias = Union[
    ParametricLifetimeModel[*tuple[_Any_Number, ...]],
    FrozenParametricModel[ParametricLifetimeModel[*tuple[_Any_Number, ...]]],
]

# a AgeReplacementModel with at least 1 arg (_Any_Real) and 0 or more args (_IntOrFloat)
class AgeReplacementModel(
    ParametricLifetimeModel[_Any_Number, *tuple[_Any_Number, ...]]
):
    # here baseline can be freeze because ParametricLifetimeModel[()] is acceptable
    baseline: _Base_Parametric_Lifetime_Model
    def __init__(self, baseline: _Base_Parametric_Lifetime_Model) -> None: ...
    def sf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def hf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def chf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def cdf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def pdf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def isf(
        self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def ppf(
        self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def median(self, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def moment(
        self, n: int, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def mean(self, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def var(self, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Number]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]: ...
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> (
        _Any_Numpy_Number
        | tuple[_Any_Numpy_Number, _Any_Numpy_Number]
        | tuple[_Any_Numpy_Number, _Any_Numpy_Bool]
        | tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]
    ): ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_Any_Number], _Any_Numpy_Number],
        a: _Any_Number,
        b: _Any_Number,
        ar: _Any_Number,
        *args: _Any_Number,
        deg: int = 10,
    ) -> _Any_Numpy_Number: ...
    def freeze(
        self, ar: _Any_Number, *args: _Any_Number
    ) -> FrozenParametricModel[
        ParametricLifetimeModel[_Any_Number, *tuple[_Any_Number, ...]]
    ]: ...

class LeftTruncatedModel(
    ParametricLifetimeModel[*tuple[_Any_Number, *tuple[_Any_Number, ...]]]
):
    baseline: _Base_Parametric_Lifetime_Model
    def __init__(self, baseline: _Base_Parametric_Lifetime_Model) -> None: ...
    def sf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def hf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def chf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def cdf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    def pdf(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def isf(
        self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def ichf(
        self, cumulative_hazard_rate: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def ppf(
        self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def median(self, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(
        self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number
    ) -> _Any_Numpy_Number: ...
    @override
    def moment(
        self, n: int, ar: _Any_Number, *args: _Any_Number
    ) -> NDArray[np.float64]: ...
    @override
    def mean(self, ar: _Any_Number, *args: _Any_Number) -> NDArray[np.float64]: ...
    @override
    def var(self, ar: _Any_Number, *args: _Any_Number) -> NDArray[np.float64]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[False] = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> _Any_Numpy_Number: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[False] = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[False] = False,
        return_entry: Literal[True] = True,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Number]: ...
    @overload
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: Literal[True] = True,
        return_entry: Literal[True] = True,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]: ...
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[
            Union[
                int, np.random.Generator, np.random.BitGenerator, np.random.RandomState
            ]
        ] = None,
    ) -> (
        _Any_Numpy_Number
        | tuple[_Any_Numpy_Number, _Any_Numpy_Number]
        | tuple[_Any_Numpy_Number, _Any_Numpy_Bool]
        | tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]
    ): ...
    @override
    def ls_integrate(
        self,
        func: Callable[[_Any_Number], _Any_Numpy_Number],
        a: _Any_Number,
        b: _Any_Number,
        ar: _Any_Number,
        *args: _Any_Number,
        deg: int = 10,
    ) -> _Any_Numpy_Number: ...
    def freeze(
        self, ar: _Any_Number, *args: _Any_Number
    ) -> FrozenParametricModel[
        ParametricLifetimeModel[_Any_Number, *tuple[_Any_Number, ...]]
    ]: ...
