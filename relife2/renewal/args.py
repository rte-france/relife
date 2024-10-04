from functools import wraps
from typing import Callable, TypedDict

import numpy as np
from numpy.typing import NDArray

# for generic TypedDict see : https://github.com/python/mypy/pull/13389 (python 3.11+)


ArgsDict = TypedDict(
    "ArgsDict",
    {
        "model": tuple[NDArray[np.float64], ...] | NDArray[np.float64],
        "initmodel": tuple[NDArray[np.float64], ...] | NDArray[np.float64],
        "reward": tuple[NDArray[np.float64], ...] | NDArray[np.float64],
        "initreward": tuple[NDArray[np.float64], ...] | NDArray[np.float64],
        "discount": tuple[NDArray[np.float64], ...] | NDArray[np.float64],
    },
    total=False,
)


def argscheck(method: Callable) -> Callable:
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        for key, value in kwargs.items():
            if "args" in key and value is not None:
                if isinstance(value, np.ndarray):
                    value = (value,)
                if self.nb_assets > 1:
                    for array in value:
                        if array.ndim != 2:
                            raise ValueError(
                                f"If nb_assets is more than 1 (got {self.nb_assets}), args array must have 2 dimensions"
                            )
                        if array.shape[0] != self.nb_assets:
                            raise ValueError(
                                f"Expected {self.nb_assets} nb assets but got {array.shape[0]} in {key}"
                            )
                else:
                    for array in value:
                        if array.ndim > 1:
                            raise ValueError(
                                f"If nb_assets is 1, args array cannot have more than 1 dimension"
                            )
        return method(self, *args, **kwargs)

    return wrapper
