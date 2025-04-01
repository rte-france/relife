from itertools import chain
from typing import Any, Iterator, Self

import numpy as np
from numpy.typing import NDArray


class ParamsTree:
    """
    Tree-structured parameters.

    Every ``ParametricModel`` are composed of ``Parameters`` instance.
    """

    def __init__(self, **kwargs):
        self._node_data = {}
        if kwargs:
            self._node_data = kwargs
        self.parent = None
        self.leaves = {}
        self._names, self._values = [], []

    @property
    def node_data(self):
        """data of current node as dict"""
        return self._node_data

    @node_data.setter
    def node_data(self, new_values: dict[str, Any]):
        self._node_data = new_values
        self.update()

    @property
    def names(self):
        """keys of current and leaf nodes as list"""
        return self._names

    @names.setter
    def names(self, new_names: list[str]):
        self.set_names(new_names)
        self.update_parents()

    @property
    def values(self):
        """values of current and leaf nodes as list"""
        return self._values

    @values.setter
    def values(self, new_values: list[Any]):
        self.set_values(new_values)
        self.update_parents()

    def set_values(self, new_values: list[Any]):
        if len(new_values) != len(self):
            raise ValueError(
                f"values expects {len(self)} items but got {len(new_values)}"
            )
        self._values = new_values
        pos = len(self._node_data)
        self._node_data.update(zip(self._node_data, new_values[:pos]))
        for leaf in self.leaves.values():
            leaf.set_values(new_values[pos : pos + len(leaf)])
            pos += len(leaf)

    def set_names(self, new_names: list[str]):
        if len(new_names) != len(self):
            raise ValueError(
                f"names expects {len(self)} items but got {len(new_names)}"
            )
        self._names = new_names
        pos = len(self._node_data)
        self._node_data = {new_names[:pos][i]: v for i, v in self._node_data.values()}
        for leaf in self.leaves.values():
            leaf.set_names(new_names[pos : pos + len(leaf)])
            pos += len(leaf)

    def __len__(self):
        return len(self._names)

    def __contains__(self, item):
        """contains only applies on current node"""
        return item in self._node_data

    def __getitem__(self, item):
        return self._node_data[item]

    def __setitem__(self, key, value):
        self._node_data[key] = value
        self.update()

    def __delitem__(self, key):
        del self._node_data[key]
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
        yield list(self._node_data.items())
        for leaf in self.leaves.values():
            yield list(chain.from_iterable(leaf.items_walk()))

    def all_items(self) -> Iterator:
        return chain.from_iterable(self.items_walk())

    def update_items(self):
        """parallel iterations : faster than update_value followed by update_keys"""
        try:
            next(self.all_items())
            _k, _v = zip(*self.all_items())
            self._names = list(_k)
            self._values = list(_v)
        except StopIteration:
            pass

    def update_parents(self):
        if self.parent is not None:
            self.parent.update()

    def update(self):
        """update names and values of current and parent nodes"""
        self.update_items()
        self.update_parents()


class Parametric:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """

    def __init__(self):
        self.params_tree = ParamsTree()
        self.leaf_models = {}

    @property
    def params(self) -> NDArray[np.float64]:
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
        return np.array(self.params_tree.values, dtype=np.float64)

    @params.setter
    def params(self, new_values: NDArray[np.float64]):
        self.params_tree.values = new_values

    @property
    def params_names(self):
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
        return self.params_tree.names

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
            if name in self.params_tree.node_data:
                raise ValueError(f"{name} already exists as param name")
            if name in self.leaf_models:
                raise ValueError(f"{name} already exists as leaf function")
        for name, module in kwcomponents.items():
            self.leaf_models[name] = module
            self.params_tree.set_leaf(f"{name}.params", module.params_tree)

    def new_params(self, **kwparams: float):
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
            if name in self.leaf_models.keys():
                raise ValueError(f"{name} already exists as function name")
        self.params_tree.node_data = kwparams

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
        if name in super().__getattribute__("leaf_models"):
            return super().__getattribute__("leaf_models")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ("params_tree", "leaf_models"):
            super().__setattr__(name, value)
        elif name in self.params_tree:
            self.params_tree[name] = value
        elif name in self.leaf_models:
            raise ValueError(
                "Can't modify leaf ParametricComponent. Recreate ParametricComponent instance instead"
            )
        else:
            super().__setattr__(name, value)
