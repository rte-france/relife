"""
This module defines Parameters class used by survival models' functions
"""

from types import EllipsisType
from typing import Self, Union

import numpy as np
from numpy.typing import ArrayLike

Index = Union[EllipsisType, int, slice, tuple[EllipsisType, int, slice, ...]]


class Parameters:
    """
    Object that encapsulates model parameters names and values
    It emulated a 1D np.array with named values

    Examples:
        >>> params = Parameters(rate=1, scale=2)
        >>> params.rate
        1.0
        >>> params.values
        array([1., 2.])
        >>> other_params = Parameters("w0", "w1", "w2")
        >>> other_params.values = (3., 5., 6.)
        >>> other_params.values
        array([3., 5., 6.])
        >>> params.append(other_params)
        >>> params.values
        array([1., 2., 3., 5., 6.])
        >>> params.w1
        5.0
    """

    def __init__(self, *names: str, **knames: float):
        for name in names:
            setattr(self, name, np.random.random())
        for name, value in knames.items():
            setattr(self, name, float(value))
        self.indice_to_name = dict(enumerate(names + tuple(knames.keys())))

    def __len__(self):
        return len(self.indice_to_name)

    @property
    def names(self) -> tuple[str, ...]:
        """
        Returns:
            int: nb of parameters (alias of len)
        """
        return tuple(self.indice_to_name.values())

    @property
    def size(self) -> int:
        """
        Returns:
            int: nb of parameters (alias of len)
        """
        return len(self)

    @property
    def values(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Parameters attributes values encapsulated in np.ndarray
        """
        return np.array(
            [getattr(self, name) for name in self.indice_to_name.values()],
            dtype=np.float64,
        )

    @values.setter
    def values(self, new_values: Union[float, ArrayLike]) -> None:
        """
        Affects new values to Parameters attributes
        Args:
            new_values (Union[float, ArrayLike]):
        """
        new_values = np.asarray(new_values, dtype=np.float64).reshape(
            -1,
        )
        nb_of_params = self.size
        if new_values.size != nb_of_params:
            raise ValueError(
                f"Can't set different number of params, expected {nb_of_params} param values, got {new_values.size}"
            )
        for indice, name in self.indice_to_name.items():
            setattr(self, name, new_values[indice])

    def __getitem__(self, index: Index) -> Union[np.float64, np.ndarray]:
        try:
            return self.values[index]
        except IndexError as exc:
            raise IndexError(
                f"Invalid index : params indices are 1d from 0 to {len(self) - 1}"
            ) from exc

    def __setitem__(
        self,
        index: Index,
        new_values: Union[float, ArrayLike],
    ) -> None:
        new_values = np.asarray(new_values, dtype=np.float64).reshape(
            -1,
        )
        indice_to_change = ~self.values.astype(bool)
        try:
            indice_to_change[index] = True
        except IndexError as exc:
            raise IndexError(
                f"Invalid index : params indices are 1d from 0 to {len(self) - 1}"
            ) from exc
        for pos, indice in enumerate(np.where(indice_to_change)[0]):
            setattr(self, self.indice_to_name[indice], np.float64(new_values[pos]))

    def append(self, params: Self) -> None:
        """
        Appends another Parameters object to itself
        Args:
            params (Parameters): Parameters object to append
        """
        if set(self.names) & set(params.names):
            raise ValueError("Can't append Parameters object having common param names")
        for name in params.names:
            setattr(self, name, getattr(params, name))
        self.indice_to_name.update(
            {indice + len(self): name for indice, name in params.indice_to_name.items()}
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
