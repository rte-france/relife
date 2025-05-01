from collections import UserDict
from itertools import chain
from typing import Any, Iterator, Optional, Self

import numpy as np
from numpy.typing import ArrayLike
from numpy.typing import NDArray


class ParametricModel:
    """
    Base class to create a parametric_model core.

    Any parametric_model core must inherit from `ParametricModel`.
    """

    def __init__(self, **kwparams: Optional[float]):
        self._parameters = Parameters(**kwparams)
        self._nested_models = {}
        self._fitting_results = None

    @property
    def params(self) -> NDArray[np.float64] | NDArray[np.complex128]:
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
        return np.array(tuple(self._parameters.allvalues()))

    @params.setter
    def params(self, values: NDArray[np.float64] | NDArray[np.complex128] | ArrayLike):
        values = np.asarray(values)
        if values.ndim > 1:
            raise ValueError(f"Expected params values to be 1d array. Got {values.ndim} ndim")
        self._parameters.set_allvalues(*values)

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
        return tuple(self._parameters.allkeys())

    @property
    def nb_params(self) -> int:
        """
        Number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return len(self._parameters)

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
            if name in self._parameters.nodedata:
                raise ValueError(f"{name} already exists as param name")
            if name in self._nested_models:
                raise ValueError(f"{name} already exists as leaf function")
        for name, model in kwcomponents.items():
            self._nested_models[name] = model
            self._parameters.set_leaf(f"{name}.params", getattr(model, "_params"))

    def new_params(self, **kwparams: Optional[float]):
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
            if name in self._nested_models.keys():
                raise ValueError(f"{name} already exists as function name")
        self._parameters.nodedata = kwparams

    def nested_models(self) -> Iterator:
        """parallel walk through key value pairs"""

        def items_walk(model: Self) -> Iterator:
            yield list(model._nested_models.items())
            for leaf in model._nested_models.values():
                yield list(chain.from_iterable(items_walk(leaf)))

        return chain.from_iterable(items_walk(self))

    def __getattr__(self, name: str):
        class_name = type(self).__name__
        if name in self.__dict__:
            return self.__dict__[name]
        if name in super().__getattribute__("_params"):
            return super().__getattribute__("_params")[name]
        if name in super().__getattribute__("_nested_models"):
            return super().__getattribute__("_nested_models")[name]
        raise AttributeError(f"{class_name} has no attribute named {name}")

    def __setattr__(self, name: str, value: Any):
        if name in ("_params", "_nested_models"):
            super().__setattr__(name, value)
        elif name in self._parameters:
            self._parameters[name] = value
        elif name in self._nested_models:
            raise ValueError(
                "ParametricModel named {name} is already set. If you want to change it, recreate a ParametricModel"
            )
        elif isinstance(value, ParametricModel):
            self.compose_with(**{name: value})
        else:
            super().__setattr__(name, value)


class Parameters:
    """
    Dict-like tree structured parameters.

    Every ``ParametricModel`` are composed of a ``Parameters`` instance.
    """
    parent : Optional[Self]

    def __init__(
        self,
        mapping: Optional[dict[str, float | complex | np.float64 | np.complex128]] = None,
        /,
        **kwargs: float | complex | np.float64 | np.complex128
    ):
        self._nodedata = {}
        if mapping is not None:
            self._nodedata = {k: np.asarray(v) if v is not None else np.nan for k, v in mapping.items()}
        if kwargs:
            self._nodedata.update(**{k: np.asarray(v) if v is not None else np.nan for k, v in kwargs.items()})

        self.parent = None
        self._leaves = {}
        self._allkeys, self._allvalues = (), ()

    def get_leaf(self, item : str):
        return self._leaves[item]

    def set_leaf(self, key : str, value : Self):
        if key not in self._leaves:
            value.parent = self
        self._leaves[key] = value
        self._update()

    def del_leaf(self, key : str):
        del self._leaves[key]
        self._update()

    @property
    def nodedata(self) -> dict[str, NDArray[np.float64] | NDArray[np.complex128]]:
        """data of current node as dict"""
        return self._nodedata

    @nodedata.setter
    def nodedata(self, mapping: dict[str, Optional[float | complex | np.float64 | np.complex128]]):
        if set(mapping.keys()).issubset(self._leaves.keys()):
            raise ValueError("Parameters names can't have the same names as leaves")
        self._nodedata = {k: np.asarray(v) if v is not None else np.nan for k, v in mapping.items()}
        self._update()

    def allkeys(self) -> Iterator[str]:
        return iter(self._allkeys)

    def allvalues(self) -> Iterator[np.float64 | np.complex128]:
        return iter(self._allvalues)

    def allitems(self) -> Iterator[tuple[str, np.float64 | np.complex128]]:
        return zip(self._allkeys, self._allvalues)

    def set_allvalues(self, *values: Optional[float | complex | np.float64 | np.complex128]):
        if len(values) != len(self):
            raise ValueError(f"values expects {len(self)} items but got {len(values)}")
        self._allvalues = tuple((np.nan if v is None else np.asarray(v) for v in values))
        pos = len(self._nodedata)
        self._nodedata.update(zip(self._nodedata, values[:pos]))
        for leaf in self._leaves.values():
            leaf.set_allvalues(*values[pos : pos + len(leaf)])
            pos += len(leaf)
        if self.parent is not None:
            self.parent._update()

    def set_allkeys(self, *keys: str):
        if len(keys) != len(self):
            raise ValueError(f"names expects {len(self)} items but got {len(keys)}")
        self._allkeys = keys
        pos = len(self._nodedata)
        self._nodedata = {keys[:pos][i]: v for i, v in self._nodedata.values()}
        for leaf in self._leaves.values():
            leaf.set_allkeys(*keys[pos : pos + len(leaf)])
            pos += len(leaf)
        if self.parent is not None:
            self.parent._update()

    def __len__(self):
        return len(self._allkeys)

    def __contains__(self, item : str):
        """contains only applies on current node"""
        return item in self._nodedata

    def __getitem__(self, item : str):
        return self._nodedata[item]

    def __setitem__(self, key : str, value : Optional[float | complex | np.number]):
        self._nodedata[key] = value
        self._update()

    def __delitem__(self, key : str):
        del self._nodedata[key]
        self._update()

    def _update(self):
        """update names and values of current and parent nodes"""
        try:
            next(self.allitems())
            _k, _v = zip(*self.allitems())
            self._allkeys = list(_k)
            self._allvalues = list(_v)
        except StopIteration:
            pass
        if self.parent is not None:
            self.parent._update()


# Use Mixin to preserve type frozen instance. Ex : FrozenParametricLifetimeModel := ParametricLifetimeModel[()]
class FrozenMixin:

    @property
    def args(self) -> tuple[float | NDArray[np.float64], ...]:
        return getattr(self, "_args", ())

    def freeze_args(self, **kwargs: float | NDArray[np.float64]):
        for name, value in kwargs.items():
            value = np.asarray(value)
            ndim = value.ndim
            if ndim > 2:
                raise ValueError(
                    f"Number of dimension can't be higher than 2. Got {ndim} for {name}"
                )
            match name:
                case "covar":
                    if ndim <= 1:
                        value = value.reshape(1, -1)
                case "a0" | "ar" | "ar1" | "cf" | "cp" | "cr":
                    if ndim == 2:
                        if value.shape[-1] > 1:
                            raise ValueError
                    value = value.reshape(-1, 1)
            if self.nb_assets != 1 and value.shape[0] not in (1, self.nb_assets):
                raise ValueError(
                    f"Frozen args have already {self.nb_assets} nb_assets values but given value has {value.shape[0]}"
                )
            setattr(self, "_args", self.args + (value,))

    @property
    def nb_assets(self) -> int:
        if bool(self.args):
            return np.broadcast_shapes(*map(np.shape, self.args))[0]
        return 1
