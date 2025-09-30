import importlib as _importlib

from relife.lifetime_model._base import is_lifetime_model
from relife.stochastic_process._base import is_stochastic_process

# only expose base routines, ParametricModel and FrozenParametricModel must be imported from relife.base explicitly
from .base import (
    get_nb_assets,
    is_frozen,
)

_submodules = [
    "data",
    "lifetime_model",
    "likelihood",
    "policy",
    "stochastic_process" "quadrature",
    "economic",
]

__all__ = _submodules + [
    # Non-modules:
    "get_nb_assets",
    "is_frozen",
    "is_lifetime_model",
    "is_stochastic_process",
]


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"relife.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'relife' has no attribute '{name}'")
