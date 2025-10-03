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
class AgeReplacementModel(ParametricLifetimeModel[_Any_Number, *tuple[_Any_Number, ...]]):
    # here baseline can be freeze because ParametricLifetimeModel[()] is acceptable
    baseline: _Base_Parametric_Lifetime_Model
    def __init__(self, baseline: _Base_Parametric_Lifetime_Model) -> None: ...
    def sf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def hf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def cdf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def pdf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def isf(self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ppf(self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def median(self, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def moment(self, n: int, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]: ...
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
    def freeze_args(
        self, ar: _Any_Number, *args: _Any_Number
    ) -> FrozenParametricModel[ParametricLifetimeModel[_Any_Number, *tuple[_Any_Number, ...]]]: ...

class LeftTruncatedModel(ParametricLifetimeModel[*tuple[_Any_Number, *tuple[_Any_Number, ...]]]):
    baseline: _Base_Parametric_Lifetime_Model
    def __init__(self, baseline: _Base_Parametric_Lifetime_Model) -> None: ...
    def sf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def hf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def chf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def cdf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    def pdf(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def isf(self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ichf(self, cumulative_hazard_rate: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def ppf(self, probability: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def median(self, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def mrl(self, time: _Any_Number, ar: _Any_Number, *args: _Any_Number) -> _Any_Numpy_Number: ...
    @override
    def moment(self, n: int, ar: _Any_Number, *args: _Any_Number) -> NDArray[np.float64]: ...
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
    ) -> tuple[_Any_Numpy_Number, _Any_Numpy_Bool, _Any_Numpy_Number]: ...
    def rvs(
        self,
        size: int,
        ar: _Any_Number,
        *args: _Any_Number,
        nb_assets: Optional[int] = None,
        return_event: bool = False,
        return_entry: bool = False,
        seed: Optional[Union[int, np.random.Generator, np.random.BitGenerator, np.random.RandomState]] = None,
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
    def freeze_args(
        self, ar: _Any_Number, *args: _Any_Number
    ) -> FrozenParametricModel[ParametricLifetimeModel[_Any_Number, *tuple[_Any_Number, ...]]]: ...


# control the model composition-chain (always LeftTruncated then AgeReplacement) thus facilitates methods workflow (eg rvs)
# the user only focuses on one function and instanciation details are hidden
# easier to retrieve unconditional baseline (avoid boilerplate codes to test every model composition orders)
# easier to test if it is ar and/or a0 conditional

class _LeftTruncatedAndAgeReplacementModel(ParametricLifetimeModel[_Any_Number, _Any_Number, *tuple[_Any_Number, ...]]):
    def __init__(self, baseline : _Base_Parametric_Lifetime_Model) -> None: ...

    def sf(self, time : _Any_Number, ar : _Any_Number, a0 : _Any_Number, *args : _Any_Number) -> _Any_Numpy_Number: ...

    def get_unconditional_model(self) -> _Base_Parametric_Lifetime_Model: ...
        # must work on frozen version too : if baseline is frozen returns frozen, else returns unfrozen baseline


def apply_condition_on(
    lifetime_model : _Base_Parametric_Lifetime_Model,
    apply_ar : Optional[bool] = None,
    apply_a0: Optional[bool] = None,
) -> Union[LeftTruncatedModel, AgeReplacementModel, _LeftTruncatedAndAgeReplacementModel, FrozenParametricModel[...]]: ...
    # at least one optinal bool arg must be set (if not error is raised)
    # values can be set to freeze model with this args


# added because it could cumbersome to always test every possibilities like :
# AgeReplacementModel or LeftTruncated(AgeReplacement), or Frozen(AgeReplacement), etc.
def is_ar_conditional(lifetime_model : _Base_Parametric_Lifetime_Model) -> bool: ...


def is_a0_conditional(lifetime_model : _Base_Parametric_Lifetime_Model) -> bool: ...
    # test either Frozen unfrozen_model or directly lifetime_model

    # usefull in some cases (RenewalProcess, Sample, etc.)
    # ex : if is_a0_conditional(first_lifetime_model):
    #            if lifetime_model.get_uncondtional_model() == lifetime_model:



##########
# EXAMPLES
##########

# survival analysis studies
# pph = ProportionalHazard(Weibull()).fit(time, covar, ...)
# model = apply_condition_on(pph, apply_ar=True)
# model.sf(time, ar, covar, ...)

# renewal process
# model = ProportionalHazard(Weibull()).fit(time, covar, ...)
# first_model = apply_condition_on(model, apply_a0=True)
# renewal_process = RenewalProcess(
#    model.freeze_args(covar),
#    first_model.freeze_args(a0, covar),
# )

# DEV ex : inside AgeReplacementPolicy constructor
# if a0 is given
# model := FrozenParametricModel[ParametricLifetimeModel[...]]
# model = apply_condition_on(model, apply_ar = True, apply_a0 = True) # model composition order is treated internally
# model = model.freeze_args(a0, ar) # order is given by _LeftTruncatedAndAgeReplacementModel type

# DEV other ex : inside Sample
#  if a0 is not None:
#     lifetime_model = apply_condition_on(lifetime_model, apply_a0=True)
#     lifetime_model = lifetime_model.freeze_args(a0)