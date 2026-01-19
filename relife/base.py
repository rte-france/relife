"""Base classes for all parametric models."""

from __future__ import annotations

import inspect
from itertools import chain
from typing import (
    Any,
    Generic,
    Iterator,
    Self,
    TypeVar,
    TypeVarTuple,
    final,
)

import numpy as np
from numpy.typing import NDArray
from typing_extensions import override

__all__ = ["ParametricModel", "FrozenParametricModel"]


@final
class _Parameters:
    """
    Dict-like tree structured of parameters.

    Every ``ParametricModel`` are composed of a ``_Parameters`` instance.
    """

    parent: Self | None
    _leaves: dict[str, _Parameters]
    _mapping: dict[str, float]
    _all_values: tuple[float, ...]
    _all_names: tuple[str, ...]

    def __init__(self, **kwargs: float | None) -> None:
        self.parent = None
        self._leaves = {}
        self._mapping = {}
        self._all_values = ()
        self._all_names = ()
        if bool(kwargs):
            self._mapping = {k: v if v is not None else np.nan for k, v in kwargs.items()}
            self.update_tree()  # update _names and _values

    @property
    def all_names(self) -> tuple[str, ...]:
        return self._all_names

    @property
    def all_values(self) -> tuple[float, ...]:
        return self._all_values

    @property
    def size(self) -> int:
        return len(self.all_values)

    def set_leaf(self, leaf_name: str, leaf: Self) -> None:
        """
        set a leaf or new leaf
        """
        if leaf_name not in self._leaves:
            leaf.parent = self
        self._leaves[leaf_name] = leaf
        self.update_tree()  # update _names and _values

    def set_all_values(self, values: tuple[float | None, ...]) -> None:
        """set values of all tree"""
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values but got {len(values)}")
        pos = len(self._mapping.items())
        self._mapping.update(zip(self._mapping.keys(), (np.nan if v is None else v for v in values[:pos])))
        self._all_values = tuple((np.nan if v is None else v for v in values))
        for leaf in self._leaves.values():
            leaf.set_all_values(values[pos : pos + leaf.size])
            pos += leaf.size
        if self.parent is not None:
            self.parent.update_tree()

    def __getitem__(self, name: str) -> float:
        try:
            return self._mapping[name]
        except KeyError:
            raise ValueError(f"Parameter {name} does not exist")

    def update_tree(self) -> None:
        """update names and values of current and parent nodes"""

        def items_walk(parameters: _Parameters) -> Iterator[tuple[str, float]]:
            yield tuple(parameters._mapping.items())
            for leaf in parameters._leaves.values():
                yield tuple(chain.from_iterable(items_walk(leaf)))

        generator = chain.from_iterable(items_walk(self))
        try:
            _k, _v = zip(*generator)
            self._all_names = _k
            self._all_values = _v
            # self._allnames, self._allvalues = zip(*generator)
        except StopIteration:
            pass
        # recursively update parent
        if self.parent is not None:
            self.parent.update_tree()


class ParametricModel:
    """
    Base class of every parametric models in ReLife.
    """

    _params: _Parameters
    _baseline_models: dict[str, ParametricModel]

    def __init__(self, **kwparams: float | None) -> None:
        self._params = _Parameters(**kwparams)
        self._baseline_models = {}

    @property
    def params(self) -> NDArray[np.float64]:
        """
        Parameters values.

        Returns
        -------
        ndarray
            Parameters values

        Notes
        -----
        If parameter values are not set, they are encoded as `np.nan` value.
        """
        return np.array(self._params.all_values)

    @params.setter
    def params(self, new_params: NDArray[np.float64]) -> None:
        if new_params.ndim > 1:
            raise ValueError(f"Expected params values to be 1d array. Got {new_params.ndim} ndim")
        self._params.set_all_values(tuple(v.item() for v in new_params))

    @property
    def params_names(self) -> tuple[str, ...]:
        """
        Parameters names.

        Returns
        -------
        list of str
            Parameters names

        Notes
        -----
        Parameters values can be requested (a.k.a. get) by their name at instance level.
        """
        return self._params.all_names

    @property
    def nb_params(self) -> int:
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return self._params.size

    def __getattr__(self, name: str) -> Any:
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_baseline_models"):
            return super().__getattribute__("_baseline_models").get(name)
        raise AttributeError(f"{type(self).__name__} has no attribute named {name}")

    @override
    def __setattr__(self, name: str, value: Any):
        # automatically add params of new baseline model
        if isinstance(value, ParametricModel):
            self._baseline_models[name] = value
            self._params.set_leaf(f"{name}.params", getattr(value, "_params"))
        super().__setattr__(name, value)


_ParametricModel_T = TypeVar("_ParametricModel_T", bound=ParametricModel)

Ts = TypeVarTuple("Ts")


class FrozenParametricModel(ParametricModel, Generic[_ParametricModel_T, *Ts]):
    """
    Class of every frozen parametric models.

    Frozen models encapsulate additional arguments values allowing to request the object without
    giving them.
    """

    _args: tuple[*Ts]
    _unfrozen_model: _ParametricModel_T

    def __init__(self, model: _ParametricModel_T, *args: *Ts):
        super().__init__()
        if np.any(np.isnan(model.params)):
            raise ValueError("Can't freeze a model with NaN params. Set params first")
        self._unfrozen_model = model
        self._args = args

    @property
    def args(self) -> tuple[*Ts]:
        return self._args

    @args.setter
    def args(self, value: tuple[*Ts]) -> None:
        if len(value) != len(self._args):
            raise ValueError
        self._args = value

    def unfreeze(self) -> _ParametricModel_T:
        return self._unfrozen_model

    @override
    def __getattr__(self, key: str) -> Any:
        frozen_type = self._unfrozen_model.__class__.__name__
        if key == "fit":
            raise AttributeError(f"Frozen model can't be fit")
        try:
            attr = getattr(self._unfrozen_model, key)
        except AttributeError:
            raise AttributeError(f"Frozen {frozen_type} has no attribute {key}")

        def wrapper(*args: Any, **kwargs: Any):
            return attr(*(*args, *self.args), **kwargs)

        if inspect.ismethod(attr):
            return wrapper
        return attr


# class FrozenParametricModel(ParametricModel, Generic[_ParametricModel_T, *Ts]):
#     """
#     Class of every frozen parametric models.

#     Frozen models encapsulate additional arguments values allowing to request the object without
#     giving them.
#     """

#     _args: tuple[*Ts]
#     _unfrozen_model: _ParametricModel_T

#     def __init__(self, model: _ParametricModel_T, *args: *Ts):
#         super().__init__()
#         if np.any(np.isnan(model.params)):
#             raise ValueError("Can't freeze a model with NaN params. Set params first")
#         self._unfrozen_model = model
#         self._args = args

#     @property
#     def args(self) -> tuple[*Ts]:
#         return self._args

#     @args.setter
#     def args(self, value: tuple[*Ts]) -> None:
#         if len(value) != len(self._args):
#             raise ValueError
#         self._args = value

#     def unfreeze(self) -> _ParametricModel_T:
#         return self._unfrozen_model

#     def __getattr__(self, key: str) -> Any:
#         frozen_type = self._unfrozen_model.__class__.__name__
#         if key == "fit":
#             raise AttributeError(f"Frozen model can't be fit")
#         try:
#             attr = getattr(self._unfrozen_model, key)
#         except AttributeError:
#             raise AttributeError(f"Frozen {frozen_type} has no attribute {key}")

#         def wrapper(*args: Any, **kwargs: Any):
#             return attr(*(*args, *self.args), **kwargs)

#         if inspect.ismethod(attr):
#             return wrapper
#         return attr
