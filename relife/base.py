from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING, Any, Optional, Self, Union, overload, Generic, TypeVarTuple

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from relife.lifetime_model import (
        AgeReplacementModel,
        FrozenAgeReplacementModel,
        FrozenLeftTruncatedModel,
        FrozenLifetimeRegression,
        FrozenParametricLifetimeModel,
        LeftTruncatedModel,
        LifetimeRegression,
        ParametricLifetimeModel,
    )
    from relife.stochastic_process import (
        FrozenNonHomogeneousPoissonProcess,
        NonHomogeneousPoissonProcess,
    )


class ParametricModel:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """
    args_names : tuple[str, ...] = ()
    _nested_models: dict[str, ParametricModel]

    def __init__(self, **kwparams: Optional[float]):
        self._params = Parameters(**kwparams)
        self._nested_models = {}

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
        return np.array(self._params.get_values())

    @params.setter
    def params(self, new_params: NDArray[np.float64]):
        if not isinstance(new_params, np.ndarray):
            raise ValueError(
                f"Incorrect params values. It must be contained in a 1d array. Got type {type(new_params)}"
            )
        if new_params.ndim > 1:
            raise ValueError(f"Expected params values to be 1d array. Got {new_params.ndim} ndim")
        self._params.set_values(tuple(v.item() for v in new_params))

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
        return self._params.get_names()

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

    def _set_nested_model(self, name: str, model: ParametricModel):
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
        self.args_names = self.args_names + model.args_names
        self._params.set_leaf(f"{name}.params", getattr(model, "_params"))

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_params").get_names():
            return super().__getattribute__("_params").get_param_value(name)
        if name in super().__getattribute__("_nested_models"):
            return super().__getattribute__("_nested_models").get(name)
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
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

    _parent: Optional[Self]
    _leaves: dict[str, Self]
    _nodemapping: dict[str, float]  # np.nan is float
    _names: tuple[str, ...]
    _values: tuple[float, ...]

    def __init__(self, **kwargs: Optional[float]):
        self._parent = None
        self._leaves = {}
        self._nodemapping = {}
        self._values = ()
        self._names = ()
        if bool(kwargs):
            self.set_node(kwargs)

    def get_names(self) -> tuple[str, ...]:
        return self._names

    def get_values(self) -> tuple[float, ...]:
        return self._values

    def get_leaf(self, name: str) -> Parameters:
        try:
            return self._leaves[name]
        except KeyError:
            raise ValueError(f"Parameters object does not have leaf parameters called {name} in its scope")

    def get_param_value(self, name: str) -> float:
        try:
            return self._nodemapping[name]
        except KeyError:
            raise ValueError(f"Parameters object does not have parameter name called {name} in its scope")

    def set_param_value(self, name: str, value: float) -> None:
        if name not in self._nodemapping:
            raise ValueError(f"Parameters object does not have parameter name called {name} in its scope")
        self._nodemapping[name] = value
        self._update_names_and_values()

    @property
    def size(self) -> int:
        return len(self._values)

    def set_node(self, mapping: dict[str, Optional[float]]) -> None:
        """
        set node dict
        """
        self._nodemapping = {k: v if v is not None else np.nan for k, v in mapping.items()}
        self._update_names_and_values()  # update _names and _values

    def set_leaf(self, leaf_name: str, leaf: Parameters) -> None:
        """
        set a leaf or new leaf
        """
        if leaf_name not in self._leaves:
            leaf._parent = self
        self._leaves[leaf_name] = leaf
        self._update_names_and_values()  # update _names and _values

    def set_values(self, values: tuple[float, ...]) -> None:
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

    def _update_names_and_values(self) -> None:
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


Args = TypeVarTuple("Args")


class FrozenParametricModel(ParametricModel, Generic[*Args]):
    def __init__(
        self,
        model : ParametricModel,
        *args : *Args,
    ):
        super().__init__()
        if np.any(np.isnan(model.params)):
            raise ValueError("You try to freeze a model with unsetted parameters. Set params first")

        self.args = dict(zip(model.args_names, args))

        self.unfrozen_model = model
        self.nb_assets = self._get_nb_assets(model)

    # TODO : store in one tuple, iterate while passing to 2D and take broadcast shape 0 axis length
    def _get_nb_assets(self, model):
        arg_nb_assets = None
        for k, v in self.args.items():
            if k == "covar":
                _arg_nb_assets = np.atleast_2d(np.asarray(v)).shape[0]
            elif k in ("a0", "ar"):
                _arg_nb_assets = np.asarray(v).size
            else:
                _arg_nb_assets = np.asarray(v).size
            if arg_nb_assets is not None and _arg_nb_assets != 1:
                if _arg_nb_assets != arg_nb_assets:
                    raise ValueError
            if arg_nb_assets is None:
                arg_nb_assets = _arg_nb_assets
        return arg_nb_assets

    @property
    def args_values(self):
        return tuple(self.args.values())

    def unfreeze(self) -> ParametricModel:
        return self.unfrozen_model


def is_frozen(model):
    pass


def is_lifetime_model(model):
    pass


def is_stochastic_process(model):
    pass


def is_age_replacement_policy(policy):
    pass


# see sklearn/base.py : return unfitted ParametricModel
# def clone(model: ParametricModel) -> ParametricModel: ...
