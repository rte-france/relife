from itertools import chain
from typing import Any, Generator, Iterator, Optional, Self

import numpy as np
from numpy.typing import NDArray

from relife._args import reshape_args


class ParametricModel:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """

    def __init__(self):
        self.params_tree = ParamsTree()
        self.leaves_of_models = {}
        self._fitting_results = None

    @property
    def params(self) -> NDArray[np.float64 | np.complex64]:
        """
        Parameters values.

        Returns
        -------
        ndarray
            Parameters values of the core

        Notes
        -----
        If parameter values are not set, they are encoded as `np.nan` value.

        Parameters can be by manually setting`params` through its setter, fitting the core if `fit` exists or
        by specifying all parameters values when the core object is initialized.
        """
        return np.array(self.params_tree.all_values)

    @params.setter
    def params(self, values: NDArray[np.float64 | np.complex64]):
        if values.ndim > 1:
            raise ValueError
        values: tuple[Optional[float], ...] = tuple(
            map(lambda x: x.item() if x != np.nan else None, values)
        )
        self.params_tree.all_values = values

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
        return tuple(self.params_tree.all_keys)

    @property
    def nb_params(self):
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return len(self.params_tree)

    def compose_with(self, **kwcomponents: Self):
        """Compose with new ``ParametricModel`` instance(s).

        This method must be seen as standard function composition exept that objects are not
        functions but group of functions (as object encapsulates functions). When you
        compose your ``ParametricModel`` instance with new one(s), the followings happen :

        - each new parameters are added to the current ``Parameters`` instance
        - each new `ParametricModel` instance is accessible as a standard attribute

        Like so, you can request new `ParametricModel` components in current `ParametricModel`
        instance while setting and getting all parameters. This is usefull when `ParametricModel`
        can be seen as a nested function (see `Regression`).

        Parameters
        ----------
        **kwcomponents : variadic named ``ParametricModel`` instance

            Instance names (keys) are followed by the instances themself (values).

        Notes
        -----
        If one wants to pass a `dict` of key-value, make sure to unpack the dict
        with `**` operator or you will get a nasty `TypeError`.
        """
        for name in kwcomponents.keys():
            if name in self.params_tree.data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaves_of_models:
                raise ValueError(f"{name} already exists as leaf function")
        for name, model in kwcomponents.items():
            self.leaves_of_models[name] = model
            self.params_tree.set_leaf(f"{name}.params", model.params_tree)

    def set_params(self, **kwparams: Optional[float]):
        """Change local parameters structure.

        This method only affects **local** parameters. `ParametricModel` components are not
        affected. This is usefull when one wants to change core parameters for any reason. For
        instance `Regression` model use `new_params` to change number of regression coefficients
        depending on the number of covariates that are passed to the `fit` method.

        Parameters
        ----------
        **kwparams : variadic named floats corresponding to new parameters

            Float names (keys) are followed by float instances (values).

        Notes
        -----
        If one wants to pass a `dict` of key-value, make sure to unpack the dict
        with `**` operator or you will get a nasty `TypeError`.
        """

        for name in kwparams.keys():
            if name in self.leaves_of_models.keys():
                raise ValueError(f"{name} already exists as function name")
        self.params_tree.data = kwparams

    # def __getattribute__(self, item):
    #     if not item.startswith("_") and not item.startswith("__"):
    #         return super().__getattribute__(item)
    #     if item in (
    #         "new_params",
    #         "compose_with",
    #         "params",
    #         "params_names",
    #         "nb_params",
    #     ):
    #         return super().__getattribute__(item)
    #     if not self._all_params_set:
    #         raise ValueError(f"Can't call {item} if one parameter value is not set")
    #     return super().__getattribute__(item)

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("params_tree"):
            return super().__getattribute__("params_tree")[name]
        if name in super().__getattribute__("leaves_of_models"):
            return super().__getattribute__("leaves_of_models")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ("params_tree", "leaves_of_models"):
            super().__setattr__(name, value)
        elif name in self.params_tree:
            self.params_tree[name] = value
        elif name in self.leaves_of_models:
            raise ValueError(
                "Can't modify leaf ParametricComponent. Recreate ParametricComponent instance instead"
            )
        else:
            super().__setattr__(name, value)


class ParamsTree:
    """
    Tree-structured parameters.

    Every ``Parametric`` are composed of ``Parameters`` instance.
    """

    def __init__(self):
        self.parent = None
        self._data = {}
        self.leaves = {}
        self._all_keys, self._all_values = (), ()
        self.dtype = float

    @property
    def data(self) -> dict[str, Optional[float | complex]]:
        """data of current node as dict"""
        return self._data

    @data.setter
    def data(self, mapping: dict[str, Optional[float | complex]]):
        self._data = mapping
        self.update()

    @property
    def all_keys(self) -> tuple[str, ...]:
        """keys of current and leaf nodes as list"""
        return self._all_keys

    @all_keys.setter
    def all_keys(self, keys: tuple[str, ...]):
        self.set_all_keys(*keys)
        self.update_parents()

    @property
    def all_values(self) -> tuple[Optional[float | complex], ...]:
        """values of current and leaf nodes as list"""
        return self._all_values

    @all_values.setter
    def all_values(self, values: tuple[Optional[float | complex], ...]):
        self.set_all_values(*values)
        self.update_parents()

    def set_all_values(self, *values: Optional[float | complex]):
        if len(values) != len(self):
            raise ValueError(f"values expects {len(self)} items but got {len(values)}")
        self._all_values = values
        pos = len(self._data)
        self._data.update(zip(self._data, values[:pos]))
        for leaf in self.leaves.values():
            leaf.set_all_values(*values[pos : pos + len(leaf)])
            pos += len(leaf)

    def set_all_keys(self, *keys: str):
        if len(keys) != len(self):
            raise ValueError(f"names expects {len(self)} items but got {len(keys)}")
        self._all_keys = keys
        pos = len(self._data)
        self._data = {keys[:pos][i]: v for i, v in self._data.values()}
        for leaf in self.leaves.values():
            leaf.set_all_keys(*keys[pos : pos + len(leaf)])
            pos += len(leaf)

    def __len__(self):
        return len(self._all_keys)

    def __contains__(self, item):
        """contains only applies on current node"""
        return item in self._data

    def __getitem__(self, item):
        return self._data[item]

    def __setitem__(self, key, value):
        self._data[key] = value
        self.update()

    def __delitem__(self, key):
        del self._data[key]
        self.update()

    def get_leaf(self, item):
        return self.leaves[item]

    def set_leaf(self, key, value):
        if key not in self.leaves:
            value.parent = self
        self.leaves[key] = value
        self.update()

    def del_leaf(self, key):
        del self.leaves[key]
        self.update()

    def items_walk(self) -> Iterator:
        """parallel walk through key value pairs"""
        yield list(self._data.items())
        for leaf in self.leaves.values():
            yield list(chain.from_iterable(leaf.items_walk()))

    def all_items(self) -> Iterator:
        return chain.from_iterable(self.items_walk())

    def update_items(self):
        """parallel iterations : faster than update_value followed by update_keys"""
        try:
            next(self.all_items())
            _k, _v = zip(*self.all_items())
            self._all_keys = list(_k)
            self._all_values = list(_v)
        except StopIteration:
            pass

    def update_parents(self):
        if self.parent is not None:
            self.parent.update()

    def update(self):
        """update names and values of current and parent nodes"""
        self.update_items()
        self.update_parents()


class FrozenParametricModel(ParametricModel):

    frozen: bool = True

    def __init__(
        self,
        model: ParametricModel,
        *args: float | NDArray[np.float64],
    ):
        super().__init__()
        self.compose_with(model=model)
        self.kwargs = reshape_args(model, *args)

    @property
    def args(self) -> tuple[float | NDArray[np.float64], ...]:
        return tuple(self.kwargs.values())

    @property
    def ndim(self) -> int:
        return max(map(lambda x: x.ndim if isinstance(x, np.ndarray) else 1, self.args), default=1)