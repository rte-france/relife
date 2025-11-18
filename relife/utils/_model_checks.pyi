from relife.base import ParametricModel

__all__ = [
    "is_frozen",
    "is_lifetime_model",
    "is_non_homogeneous_poisson_process",
]

def is_frozen(model: ParametricModel) -> bool: ...
def is_lifetime_model(model: ParametricModel) -> bool: ...
def is_non_homogeneous_poisson_process(model: ParametricModel) -> bool: ...
