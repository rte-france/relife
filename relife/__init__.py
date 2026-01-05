from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("relife")
except PackageNotFoundError:
    # package is not installed
    pass

__all__: list[str] = []

_submodules = [
    "data",
    "lifetime_model",
    "likelihood",
    "policy",
    "stochastic_process",
    "quadrature",
    "economic",
    "typing",
    "utils",
]

__all__ += _submodules
