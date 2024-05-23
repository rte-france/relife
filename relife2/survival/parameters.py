"""
This module defines Parameters class used by survival models' functions
"""

from typing import Iterable, Self, Tuple, TypeAlias, Union

import numpy as np
from numpy.typing import ArrayLike

Index: TypeAlias = Union[Ellipsis, int, slice, Tuple[Ellipsis, int, slice]]


class Parameters:
    """
    Object that encapsulates model parameters names and values
    It emulated a 1D np.array with named values

    Examples:
        >>> params = Parameters("rate")
        >>> params.rate = 1.
        >>> params.values
        array([1.])
        >>> covar_params = Parameters("w0", "w1", "w2")
        >>> covar_params.values = (3., 5., 6.)
        >>> covar_params.values
        array([3., 5., 6.])
        >>> params.append(covar_params)
        >>> params.values
        array([1., 3., 5., 6.])
        >>> params.w1
        5.0
    """

    def __init__(self, *names: str, **knames: float):
        for name in names:
            setattr(self, name, np.float64(np.random.random()))
        for name, value in knames.items():
            setattr(self, name, np.float64(value))
        self.indice_to_name = dict(enumerate(names + tuple(knames.keys())))

    def __len__(self):
        return len(self.indice_to_name)

    @property
    def size(self) -> int:
        """
        Returns:
            int: nb of parameters (alias of len)
        """
        values = self.values
        if isinstance(values, np.int64):
            size = 1
        else:
            size = values.size
        return size

    @property
    def values(self) -> Union[np.int64, np.ndarray]:
        """
        Returns:
            np.ndarray: Parameters attributes values encapsulated in np.ndarray
        """
        if len(self.indice_to_name) == 1:
            values = getattr(self, self.indice_to_name[0])
        else:
            values = np.array(
                [getattr(self, name) for name in self.indice_to_name.values()],
                dtype=np.float64,
            )
        return values

    @values.setter
    def values(self, new_values: Union[float, ArrayLike]) -> None:
        """
        Affects new values to Parameters attributes
        Args:
            new_values (Iterable[float]):
        """
        new_values = np.asarray(new_values, dtype=np.float64).reshape(
            -1,
        )
        param_size = self.size
        if new_values.size != param_size:
            raise ValueError(
                f"Can't set different number of params, expected {param_size} param values, got {new_values.size}"
            )
        for indice, name in self.indice_to_name.items():
            setattr(self, name, new_values[indice])

    def __getitem__(self, index: Index) -> Union[np.int64, np.ndarray]:
        try:
            return self.values[index]
        except IndexError:
            raise IndexError("Invalid index values")

    def __setitem__(
        self,
        index: Index,
        new_values: Union[float, ArrayLike],
    ) -> None:
        new_values = np.asarray(new_values, dtype=np.float64).reshape(
            -1,
        )
        changed_values = ~self.values.astype(bool)
        try:
            changed_values[index] = True
        except IndexError:
            raise IndexError("Invalid index values")
        for pos, indice in enumerate(np.where(changed_values)[0]):
            setattr(self, self.indice_to_name[indice], np.float64(new_values[pos]))

    def append(self, params: Self) -> None:
        """
        Appends another Parameters object to itself
        Args:
            params (Parameters): Parameters object to append
        """
        if set(self.indice_to_name.values()) & set(params.indice_to_name.values()):
            raise ValueError("Can't append Parameters object having common param names")
        for name in params.indice_to_name.values():
            setattr(self, name, getattr(params, name))
        self.indice_to_name.update(
            {pos + len(self): name for pos, name in params.indice_to_name.items()}
        )

    def __repr__(self):
        class_name = type(self).__name__
        attributes = ", ".join(
            [f"{name}={getattr(self, name)}" for name in self.indice_to_name.values()]
        )

        return f"{class_name}({attributes})"

    def __str__(self):
        attributes = "\n".join(
            [f"{name}: {getattr(self, name)}" for name in self.indice_to_name.values()]
        )
        return f"Parameters\n---\n{attributes}"
