from itertools import chain
from typing import Self

import numpy as np


class ParametricModel:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """

    def __init__(self, **kwparams):
        self._params = Parameters(**kwparams)
        self._nested_models = {}

    @property
    def params(self):
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
        return np.array(self._params.get_values())

    @params.setter
    def params(self, new_params):
        if not isinstance(new_params, np.ndarray):
            raise ValueError(
                f"Incorrect params values. It must be contained in a 1d array. Got type {type(new_params)}"
            )
        if new_params.ndim > 1:
            raise ValueError(f"Expected params values to be 1d array. Got {new_params.ndim} ndim")
        self._params.set_values(tuple(v.item() for v in new_params))

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
        return self._params.get_names()

    @property
    def nb_params(self):
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return self._params.size

    def _set_nested_model(self, name, model):
        """Compose with new ``ParametricModel`` instance(s).

        This method acts like function. When you set a nested model instance, the followings happen :
        - each new parameters are added to the current ``Parameters`` instance
        - each new `ParametricModel` instance is accessible as a standard attribute

        Parameters
        ----------
        name : str
            model name
        model : ParametricModel
            model instance
        """
        self._nested_models[name] = model
        self._params.set_leaf(f"{name}.params", getattr(model, "_params"))

    def __getattr__(self, name):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_params").get_names():
            return super().__getattribute__("_params").get_param_value(name)
        if name in super().__getattribute__("_nested_models"):
            return super().__getattribute__("_nested_models").get(name)
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name, value):
        if name in ("_params", "_nested_models"):
            super().__setattr__(name, value)
        elif name in self._params.get_names():
            self._params.set_param_value(name, value)
        elif isinstance(value, ParametricModel):
            self._set_nested_model(name, value)
        else:  # just set a new attribute
            super().__setattr__(name, value)


# MAYBE, custom array container can replace it : https://numpy.org/doc/stable/user/basics.dispatch.html#writing-custom-array-containers
class Parameters:
    """
    Dict-like tree structured parameters.

    Every ``ParametricModel`` are composed of a ``Parameters`` instance.
    """

    def __init__(self, **kwargs):
        self._parent = None
        self._leaves = {}
        self._nodemapping = {}
        self._values = ()
        self._names = ()
        if bool(kwargs):
            self.set_node(kwargs)

    def get_names(self):
        return self._names

    def get_values(self):
        return self._values

    def get_leaf(self, name: str):
        try:
            return self._leaves[name]
        except KeyError:
            raise ValueError(f"Parameters object does not have leaf parameters called {name} in its scope")

    def get_param_value(self, name):
        try:
            return self._nodemapping[name]
        except KeyError:
            raise ValueError(f"Parameters object does not have parameter name called {name} in its scope")

    def set_param_value(self, name, value):
        if name not in self._nodemapping:
            raise ValueError(f"Parameters object does not have parameter name called {name} in its scope")
        self._nodemapping[name] = value
        self._update_names_and_values()

    @property
    def size(self):
        return len(self._values)

    def set_node(self, mapping):
        """
        set node dict
        """
        self._nodemapping = {k: v if v is not None else np.nan for k, v in mapping.items()}
        self._update_names_and_values()  # update _names and _values

    def set_leaf(self, leaf_name, leaf):
        """
        set a leaf or new leaf
        """
        if leaf_name not in self._leaves:
            leaf._parent = self
        self._leaves[leaf_name] = leaf
        self._update_names_and_values()  # update _names and _values

    def set_values(self, values):
        """set values of all tree"""
        if len(values) != self.size:
            raise ValueError(f"Expected {self.size} values but got {len(values)}")
        pos = len(self._nodemapping)
        self._nodemapping.update(zip(self._nodemapping.keys(), (np.nan if v is None else v for v in values[:pos])))
        self._values = tuple((np.nan if v is None else v for v in values))
        for leaf in self._leaves.values():
            leaf.set_values(values[pos : pos + leaf.size])
            pos += leaf.size
        if self._parent is not None:
            self._parent._update_names_and_values()

    def _update_names_and_values(self):
        """update names and values of current and parent nodes"""

        def items_walk(parameters: Self):
            yield tuple(parameters._nodemapping.items())
            for leaf in parameters._leaves.values():
                yield tuple(chain.from_iterable(items_walk(leaf)))

        generator = chain.from_iterable(items_walk(self))
        try:
            self._names, self._values = zip(*generator)
        except StopIteration:
            pass
        # recursively update parent
        if self._parent is not None:
            self._parent._update_names_and_values()


def get_nb_assets(*args):
    if not bool(args):
        return 1
    args_2d = tuple((np.atleast_2d(arys) for arys in args))

    try:
        broadcast_shape = np.broadcast_shapes(*(ary.shape for ary in args_2d))
        nb_assets = broadcast_shape[0]
    except ValueError:
        raise ValueError("Given args have incompatible shapes")
    return nb_assets


class FrozenParametricModel(ParametricModel):
    def __init__(self, model, *args):
        super().__init__()
        if np.any(np.isnan(model.params)):
            raise ValueError("You try to freeze a model with unsetted parameters. Set params first")
        self.nb_assets = get_nb_assets(*args)
        self.args = args
        self.unfrozen_model = model

    def unfreeze(self):
        return self.unfrozen_model


def is_frozen(model):
    return isinstance(model, FrozenParametricModel)


def is_lifetime_model(model):
    from relife.lifetime_model import (
        NonParametricLifetimeModel,
        ParametricLifetimeModel,
    )

    return isinstance(model, (ParametricLifetimeModel, NonParametricLifetimeModel))


def is_stochastic_process(model):
    from relife.stochastic_process import StochasticProcess

    return isinstance(model, StochasticProcess)


# see sklearn/base.py : return unfitted ParametricModel
# def clone(model: ParametricModel) -> ParametricModel: ...
