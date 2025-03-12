from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray
from relife.types import Args


def np_at_least(nb_assets: int):
    if nb_assets == 1:
        return np.atleast_1d
    else:
        return np.atleast_2d


def get_nb_assets(args_tuple: tuple[Args, ...]) -> int:
    def as_2d():
        for x in args_tuple:
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            if len(x.shape) > 2:
                raise ValueError
            yield np.atleast_2d(x)

    return max(map(lambda x: x.shape[0], as_2d()), default=1)


class NbAssets:

    def __init__(self, compute_from: Optional[tuple[str]] = None):
        self.compute_from = compute_from
        self.cached_value = None if compute_from is not None else 1

    def __get__(self, obj, objtype=None):
        if self.cached_value is not None:
            return self.cached_value
        else:
            if self.compute_from is not None:
                nb_assets = get_nb_assets(
                    tuple((getattr(obj, attr) for attr in self.compute_from))
                )
                self.cached_value = nb_assets
                return nb_assets


class ShapedArgs:

    def __init__(self, astuple: bool = False):
        self.astuple = astuple

    def __set_name__(self, owner, name):
        self.private_name = "_" + name
        self.public_name = name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(
        self, obj, value: Optional[Union[Args, tuple[Args, ...]]]
    ) -> tuple[Args, ...]:
        """
        if nb_assets is 1, values are all 1d array (event floats)
        if nb_assets is more than 1, values are all 2d array
        """
        if value is None:
            setattr(obj, self.private_name, value)
        else:
            try:
                if isinstance(value, (list, tuple)):
                    value = [
                        np_at_least(obj.nb_assets)(np.asarray(v, dtype=np.float64))
                        for v in value
                    ]
                else:
                    value = [
                        np_at_least(obj.nb_assets)(np.asarray(value, dtype=np.float64))
                    ]
            except ValueError:
                raise ValueError("Incompatible args type. Must be ArrayLike")

            if bool(value):  # if not empty
                current_nb_assets = max(
                    map(lambda x: x.shape[0] if x.ndim > 1 else 1, value), default=1
                )
                nb_assets = getattr(obj, "nb_assets")
                if current_nb_assets > nb_assets:
                    raise ValueError(
                        f"Uncorrect arg shape (nb of assets up to {current_nb_assets}) but nb_assets is set {nb_assets}"
                    )

                for i, arg in enumerate(value):
                    if arg.shape[0] == 1 and nb_assets != 1:
                        value[i] = np.tile(arg, (nb_assets, 1))
                    if arg.shape[0] != 1 and arg.shape[0] != nb_assets:
                        raise ValueError(
                            f"Args {self.public_name} shapes are inconsistent"
                        )
                    if arg.ndim > 2:
                        raise ValueError(
                            f"Args {self.public_name} can't have more than 2 dimensions"
                        )

            if self.astuple:
                setattr(obj, self.private_name, tuple(value))
            else:
                if len(value) > 1:
                    raise ValueError(
                        "If astuple is False, args can't be a sequence, set astuple to True"
                    )
                setattr(obj, self.private_name, value[0])
