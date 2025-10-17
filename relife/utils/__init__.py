from ._array_api import (
    flatten_if_possible,
    get_args_nb_assets,
    reshape_1d_arg,
)
from ._model_checks import (
    is_frozen,
    is_lifetime_model,
    is_non_homogeneous_poisson_process,
)

__all__ = [
    "get_args_nb_assets",
    "is_frozen",
    "is_lifetime_model",
    "is_non_homogeneous_poisson_process",
    "reshape_1d_arg",
    "flatten_if_possible",
]
