"""
This module defines Parameters class used by survival models' functions
"""

from collections.abc import Iterable
from random import random
from typing import Self, Union

import numpy as np


class Parameters:
    """
    Object that encapsulates model parameters names and values

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
        5.
    """

    def __init__(self, *names: str):
        for name in names:
            setattr(self, name, random())
        self._pos_to_name = dict(enumerate(names))

    def __len__(self):
        return len(self._pos_to_name)

    @property
    def values(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: Parameters attributes values encapsulated in np.ndarray
        """
        return np.array([getattr(self, name) for name in self._pos_to_name.values()])

    @values.setter
    def values(self, new_values: Iterable[float]) -> None:
        """
        Affects new values to Parameters attributes
        Args:
            new_values (Iterable[float]):
        """
        if len(new_values) != len(self._pos_to_name):
            raise ValueError("Can't set different number of param to Parameters object")
        for pos, name in self._pos_to_name.items():
            setattr(self, name, new_values[pos])

    def __getitem__(self, item: Union[int, Iterable[int], str, Iterable[str]]):
        if isinstance(item, int):
            return getattr(self, self._pos_to_name[item])
        if isinstance(item, str):
            return getattr(self, item)
        if isinstance(item, Iterable):
            return np.array([self.__getitem__(mini_item) for mini_item in item])

    def __setitem__(
        self,
        item: Union[int, Iterable[int], str, Iterable[str]],
        value: Union[float, Iterable[float]],
    ):
        if isinstance(item, int):
            setattr(self, self._pos_to_name[item], value)
        elif isinstance(item, str):
            setattr(self, item, value)
        elif isinstance(item, Iterable):
            for pos, mini_item in enumerate(item):
                self.__setitem__(mini_item, value[pos])

    def append(self, params: Self) -> None:
        """
        Appends another Parameters object to itself
        Args:
            params (Parameters): Parameters object to append
        """
        if set(self._pos_to_name.values()) & set(params._pos_to_name.values()):
            raise ValueError(
                "Can't append two Parameters object having the common param names"
            )
        for name in params._pos_to_name.values():
            setattr(self, name, getattr(params, name))
        self._pos_to_name.update(
            {pos + len(self): name for pos, name in params._pos_to_name.items()}
        )

    def __repr__(self):
        class_name = type(self).__name__
        attributes = ", ".join(
            [f"{name}={getattr(self, name)}" for name in self._pos_to_name.values()]
        )

        return f"{class_name}({attributes})"

    def __str__(self):
        attributes = "\n".join(
            [f"{name}: {getattr(self, name)}" for name in self._pos_to_name.values()]
        )
        return f"Parameters\n---\n{attributes}"
