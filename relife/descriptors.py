from typing import Union, Optional

import numpy as np
from numpy.typing import NDArray
from relife.types import ModelArgs


class ShapedArgs:

    def __set_name__(self, owner, name):
        self.private_name = "_" + name
        self.public_name = name

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name)

    def __set__(
        self, obj, value: Optional[Union[NDArray[np.float64], ModelArgs]]
    ) -> ModelArgs:
        if value is None:
            setattr(obj, self.private_name, value)
        else:
            if isinstance(value, (list, tuple)):
                args_2d = [np.atleast_2d(arg) for arg in value]
            elif isinstance(value, np.ndarray):
                args_2d = [np.atleast_2d(value)]
            elif isinstance(value, float):
                args_2d = [np.array([[value]])]
            else:
                raise ValueError(
                    f"Args {self.public_name} can be either sequence of ndarray, ndarray or float"
                )

            current_nb_assets = max(map(lambda x: x.shape[0], args_2d))
            nb_assets = getattr(obj, "nb_assets")
            if current_nb_assets > nb_assets:
                raise ValueError(
                    f"""
                    Nb assets is {nb_assets} but {self.public_name} seems to have {current_nb_assets}.
                    Set correct nb_assets (default 1) or modify {self.public_name}
                    """
                )

            for i, arg in enumerate(args_2d):
                if arg.shape[0] == 1:
                    args_2d[i] = np.tile(arg, (nb_assets, 1))
                if arg.shape[0] != 1 and arg.shape[0] != nb_assets:
                    raise ValueError(f"Args {self.public_name} shapes are inconsistent")
                if arg.ndim > 2:
                    raise ValueError(
                        f"Args {self.public_name} can't have more than 2 dimensions"
                    )

            setattr(obj, self.private_name, tuple(args_2d))
