from importlib.metadata import PackageNotFoundError, version

from .utils import get_args_nb_assets, is_frozen

# ParametricModel and FrozenParametricModel must be imported from relife.base explicitly


try:
    __version__ = version("relife")
except PackageNotFoundError:
    # package is not installed
    pass

_submodules = [
    "data",
    "lifetime_model",
    "likelihood",
    "policy",
    "stochastic_process",
    "quadrature",
    "economic",
]

__all__ = _submodules + [
    # Non-modules:
    "get_nb_assets",
    "is_frozen",
]
