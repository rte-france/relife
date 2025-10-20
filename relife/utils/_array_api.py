import numpy as np


def reshape_1d_arg(arg):
    """
    Reshapes ReLife arguments that are expected to be 0d or 1d.

    Parameters
    ----------
    arg : float, 0d or 1d array

    Returns
    -------
    np.float64 or (m, 1) shaped array
        Reshaped array used to ensure broadcasting compatibility in computations.
    """
    arg = np.float64(arg) if isinstance(arg, (float, int)) else np.asarray(arg)
    if arg.ndim == 1:
        arg = arg.reshape(-1, 1)
    elif arg.ndim > 2:
        raise ValueError("args can't be more than 2d")
    return arg

def flatten_if_possible(value):
    """
    Flatten array-like object when possible.

    Parameters
    ----------
    value : np.ndarray

    Returns
    -------
    np.ndarray
        Flattened array.
    """
    if value.ndim != 0:
        return value.flatten()
    return value

def get_args_nb_assets(*args):
    """
    Gets the number of assets encoded in args.
    """
    if not bool(args):
        return 1
    reshaped_args = tuple((np.atleast_2d(arg) for arg in args))
    try:
        broadcast_shape = np.broadcast_shapes(*(ary.shape for ary in reshaped_args))
    except ValueError:
        raise ValueError("args have incompatible shapes")
    if len(broadcast_shape) == 0:
        return 1
    return broadcast_shape[0]
