from functools import wraps
from typing import Union

import numpy as np
from numpy.typing import ArrayLike


def array_factory(obj: Union[ArrayLike, None], nb_units=None) -> np.ndarray:
    """
    Converts object input to 2d array of shape (n, p)
    n is the number of units
    p is the number of points
    Args:
        nb_units ():
        obj: object input

    Returns:
        np.ndarray: 2d array
    """
    if obj is not None:
        try:
            obj = np.asarray(obj, dtype=np.float64)
        except Exception as error:
            raise ValueError("Invalid type of param") from error
        if obj.ndim == 0:
            obj = obj.reshape(-1, 1)
        elif obj.ndim == 1:
            if nb_units is None:
                obj = obj.reshape(-1, 1)
            else:
                obj = obj.reshape(nb_units, -1)
        elif obj.ndim > 2:
            raise ValueError(
                f"input FloatArray can't have more than 2 dimensions : got {obj}"
            )
    return obj


def transform_vars_to_valid_array(*mainvars: ArrayLike, **extravars: ArrayLike):
    try:
        mainvars = [array_factory(var) for var in mainvars]
    except ValueError as err:
        raise ValueError(f"Invalid extravars type or shape") from err
    nb_units = [var.shape[0] for var in mainvars if var is not None]
    if bool(nb_units):
        if len(set(nb_units)) != 1:
            raise ValueError("all args must have the same number of units")
        nb_units = mainvars[0].shape[0]
    else:
        nb_units = None
    for name, extravar in extravars.items():
        try:
            extravar = array_factory(extravar, nb_units=nb_units)
        except ValueError as err:
            raise ValueError(
                f"Invalid extravars {name} type or shape. Change type to ArrayLike or give it compatible shape to meet {nb_units} nb of units"
            ) from err
        extravars[name] = extravar
    return mainvars, extravars


def are_all_extravars_given(functions, **extravars: ArrayLike):
    given_extravars = set(extravars.keys())
    expected_extravars = set(functions.extravars_names)
    common_extravars = given_extravars & expected_extravars
    if common_extravars != set(expected_extravars):
        raise ValueError(
            f"Method expects {expected_extravars} but got only {common_extravars}"
        )


def preprocess_vars(functions, *mainvars: ArrayLike, **extravars: ArrayLike):
    are_all_extravars_given(functions, **extravars)
    mainvars, extravars = transform_vars_to_valid_array(*mainvars, **extravars)
    if bool(mainvars):
        return *mainvars, extravars
    else:
        return extravars


def squeeze(method):
    """
    Args:
        method ():

    Returns:
    """

    @wraps(method)
    def _squeeze(self, *args, **kwargs):
        method_output = method(self, *args, **kwargs)
        return np.squeeze(method_output)[()]

    return _squeeze
