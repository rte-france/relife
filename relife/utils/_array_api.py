import numpy as np


def reshape_1d_arg(arg):
    """
    Reshapes given arg that can be 1d max to either () or (nb_assets, 1)
    It concerns ar, a0, etc.
    If arg is 1d, shape is (nb_assets, 1)
    If arg is a scalar, shape is ()
    """
    arg = np.float64(arg) if isinstance(arg, (float, int)) else np.asarray(arg)
    if arg.ndim == 1:
        arg = arg.reshape(-1, 1)
    elif arg.ndim > 2:
        raise ValueError("args can't be more than 2d")
    return arg


# def to_relife_shape_2d_arg(arg):
#     """
#     Reshapes given arg that can be 2d max to either () or (nb_assets, n)
#     It concerns covar.
#     If arg is 1d, shape is (nb_assets, 1)
#     If arg is a scalar, shape is ()
#     """
#     arg = np.float64(arg) if isinstance(arg, (float, int)) else np.asarray(arg)
#     if arg.ndim == 1:
#         arg = arg.reshape(-1, 1)
#     elif arg.ndim > 2:
#         raise ValueError("args can't be more than 2d")
#     return arg


def get_args_nb_assets(*args):
    """
    Computes the number of assets encoded in args.
    It broadcasts shapes of args. The broadcasted shape can be (), (m, 1) or (m, n).
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


def filter_nonetype_args(*args):
    """
    Removes None args
    """
    filtered_args = ()
    for arg in args:
        if arg is not None:
            filtered_args += (arg,)
    return filtered_args
