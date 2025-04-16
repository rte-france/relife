import copy
import functools

import numpy as np
from numpy.typing import NDArray


def _reshape_like(
    arg_value: float | NDArray[np.float64], arg_name: str, nb_assets: int
):
    arg_value = np.asarray(arg_value)
    ndim = arg_value.ndim
    if ndim > 2:
        raise ValueError(
            f"Number of dimension can't be higher than 2. Got {ndim} for {arg_name}"
        )
    match arg_name:
        case "ar" | "ar1":
            if arg_value.ndim <= 1:
                if arg_value.size == 1:
                    return arg_value.item()
            else:
                arg_value = arg_value.reshape(-1, 1)
                if nb_assets != 1:
                    if arg_value.shape[0] != nb_assets:
                        raise ValueError(
                            f"Invalid {arg_name} shape. Got {nb_assets} nb assets but got {arg_value.shape} {arg_name} shape"
                        )
            return arg_value


def get_if_none(*args_names: str):
    """
    Decorators that get the attribute value if argument value is None
    Reshape depending on number of assets.
    If both are None, an error is raised.
    Priority is always given to the attribute value

    Parameters
    ----------
    args_names

    Returns
    -------

    """

    def decorator(method):
        @functools.wraps(method)
        def wrapper(self, *args, **kwargs):
            new_kwargs = copy.deepcopy(kwargs)
            for name in args_names:
                attr_value = getattr(self, name)
                arg_value = kwargs.get(name, None)
                if attr_value is None and arg_value is None:
                    if name == "ar1" and self.model1 is not None:
                        raise ValueError(
                            f"{name} is not set. If fit exists, you may need to fit the object first or instanciate the object with {name}"
                        )
                elif attr_value is not None and arg_value is not None:
                    # priority on arg
                    new_kwargs[name] = _reshape_like(
                        arg_value, name, self.model_instances.nb_assets
                    )
                elif attr_value is None and arg_value is not None:
                    # priority on argue)
                    new_kwargs[name] = _reshape_like(
                        arg_value, name, self.model_instances.nb_assets
                    )
                elif attr_value is not None and arg_value is None:
                    new_kwargs[name] = _reshape_like(
                        attr_value, name, self.model_instances.nb_assets
                    )
                else:
                    new_kwargs[name] = _reshape_like(
                        arg_value, name, self.model_instances.nb_assets
                    )
            return method(self, *args, **new_kwargs)

        return wrapper

    return decorator
